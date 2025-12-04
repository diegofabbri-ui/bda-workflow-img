# app.py — BD&A Workflow API (gpt-5-chat-latest, conversazione unica C1–C3)
# C1–C3: UNA sola chiamata a OpenAI (Responses → fallback Chat Completions)
# C4: generato in locale dal backend usando il prompt filtrato da C3
# Immagini: endpoint separato /api/generate-image (gpt-image-1)

import os
import json
import base64
import mimetypes
import logging
from io import BytesIO
from datetime import datetime
from typing import Optional, Tuple, Dict

from fastapi import FastAPI, Form, HTTPException, Request
from fastapi.responses import JSONResponse, HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

# OpenAI SDK
try:
    from openai import OpenAI, RateLimitError, APIStatusError, BadRequestError
except Exception:  # se la libreria non è installata
    OpenAI = None
    RateLimitError = Exception
    APIStatusError = Exception
    BadRequestError = Exception

# Pillow (opzionale per resize immagini)
try:
    from PIL import Image
    PIL_AVAILABLE = True
except Exception:
    Image = None
    PIL_AVAILABLE = False

# ────────────────────────────────────────────────────────────────────────────────
# Logging
# ────────────────────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("bda-workflow")

# ────────────────────────────────────────────────────────────────────────────────
# App & CORS
# ────────────────────────────────────────────────────────────────────────────────
app = FastAPI(title="BD&A Workflow API — gpt-5-chat-latest (single conversation)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # frontend Aruba, localhost, ngrok, ecc.
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOADS_DIR = os.path.join(BASE_DIR, "uploads")
GENERATED_DIR = os.path.join(UPLOADS_DIR, "generated")
STATIC_DIR = os.path.join(BASE_DIR, "static")
os.makedirs(UPLOADS_DIR, exist_ok=True)
os.makedirs(GENERATED_DIR, exist_ok=True)

# ────────────────────────────────────────────────────────────────────────────────
# Config
# ────────────────────────────────────────────────────────────────────────────────
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Modelli (tutti configurabili via env)
OPENAI_MODEL_CHAT = os.getenv("OPENAI_MODEL_CHAT", "gpt-5-chat-latest")      # Responses
OPENAI_MODEL_VISION = os.getenv("OPENAI_MODEL_VISION", "gpt-5-chat-latest")  # Responses vision
OPENAI_MODEL_FALLBACK = os.getenv("OPENAI_MODEL_FALLBACK", "gpt-4o-mini")    # Chat Completions fallback
OPENAI_IMAGE_MODEL = os.getenv("OPENAI_IMAGE_MODEL", "gpt-image-1")
IMAGE_API_ENABLED = os.getenv("IMAGE_API_ENABLED", "true").lower() == "true"

# Token budget per l'intero workflow C1–C3
MAX_TOKENS_WORKFLOW = int(os.getenv("MAX_TOKENS_WORKFLOW", "1200"))

# Template (fallback interni se file mancanti)
PROMPT_FILES = {
    "fase1": os.path.join(BASE_DIR, "PROMPT-1-X-IMG.txt"),
    "fase2": {
        "gancio": os.path.join(BASE_DIR, "GANCIO.txt"),
        "intermedia": os.path.join(BASE_DIR, "INTERMEDIA.txt"),
        "cta": os.path.join(BASE_DIR, "CTA.txt"),
    },
    "fase3": {
        "BD&A": os.path.join(BASE_DIR, "PROMPT-3-BD-A.txt"),
        "Sportello Immobiliare": os.path.join(BASE_DIR, "PROMPT-3-SI.txt"),
        "Sportello del Cittadino": os.path.join(BASE_DIR, "PROMPT-3-SDC.txt"),
    },
    "fase4": os.path.join(BASE_DIR, "PROMPT-4-X-IMG.txt"),
}

DEFAULT_TEMPLATES = {
    "fase1": (
        "Analizza il brand {{azienda}} per il progetto {{progetto}}. "
        "Formato: {{formato}}. Tipo slide: {{tipo_slide}}. Copy: {{copy}}. "
        "Se presente immagine di riferimento, usala per il contesto visivo."
    ),
    "fase2": {
        "gancio": (
            "Ottimizza il copy per gancio breve e d’impatto per {{azienda}}. "
            "Copy di partenza: {{copy}}."
        ),
        "intermedia": (
            "Sviluppa corpo intermedio chiaro e sintetico per {{azienda}}. "
            "Copy base: {{copy}}."
        ),
        "cta": (
            "Scrivi una CTA implicita e persuasiva per {{azienda}}. "
            "Copy base: {{copy}}."
        ),
    },
    "fase3": {
        "BD&A": (
            "Genera il PROMPT dettagliato per immagine del brand BD&A. "
            "Inizia con 'Prompt' e descrivi la scena coerente con {{copy}}."
        ),
        "Sportello Immobiliare": (
            "Genera il PROMPT dettagliato per immagine del brand Sportello Immobiliare. "
            "Inizia con 'Prompt' e descrivi la scena legata a {{copy}}."
        ),
        "Sportello del Cittadino": (
            "Genera il PROMPT dettagliato per immagine del brand Sportello del Cittadino. "
            "Inizia con 'Prompt' e descrivi la scena legata a {{copy}}."
        ),
    },
    "fase4": (
        "PROMPT 4 X IMG\n\n"
        "Azienda: {{azienda}}\n"
        "Progetto: {{progetto}}\n"
        "Tipo slide: {{tipo_slide}}\n"
        "Copy: {{copy}}\n\n"
        "PROMPT generato:\n"
        "{{prompt_generato}}\n"
    ),
}

# Reference auto per azienda (se il file manca → C1 solo testo)
IMMAGINI_AZIENDA = {
    "BD&A": "WhatsApp-Image-2025-10-16-at-10.32.55-PM.jpg",
    "Sportello del Cittadino": "WhatsApp-Image-2025-10-28-at-7.13.01-PM.jpg",
    "Sportello Immobiliare": "Gemini_Generated_Image_nwkerxnwkerxnwke.jpg",
}
SAFE_IMAGE_MIMES = {"image/png", "image/jpeg", "image/webp"}

# ────────────────────────────────────────────────────────────────────────────────
# Utils
# ────────────────────────────────────────────────────────────────────────────────
def ensure_openai():
    if not OPENAI_API_KEY or OpenAI is None:
        raise HTTPException(status_code=500, detail="OpenAI SDK o OPENAI_API_KEY mancante.")
    return OpenAI(api_key=OPENAI_API_KEY, timeout=60.0)


def leggi_file(path: str, fallback: Optional[str] = None) -> str:
    if path and os.path.exists(path):
        try:
            with open(path, encoding="utf-8") as f:
                return f.read()
        except Exception:
            pass
    return fallback or ""


def template_for(stage: str, key: Optional[str] = None) -> str:
    if stage == "fase1":
        return leggi_file(PROMPT_FILES["fase1"], DEFAULT_TEMPLATES["fase1"])
    if stage == "fase2" and key in PROMPT_FILES["fase2"]:
        return leggi_file(PROMPT_FILES["fase2"][key], DEFAULT_TEMPLATES["fase2"].get(key, ""))
    if stage == "fase3" and key in PROMPT_FILES["fase3"]:
        return leggi_file(PROMPT_FILES["fase3"][key], DEFAULT_TEMPLATES["fase3"].get(key, ""))
    if stage == "fase4":
        return leggi_file(PROMPT_FILES["fase4"], DEFAULT_TEMPLATES["fase4"])
    return ""


def compila_template(template: str, mapping: Dict[str, str]) -> str:
    out = template
    for k, v in mapping.items():
        out = out.replace(f"{{{{{k}}}}}", str(v))
    return out


def estrai_da_prompt(testo: str) -> str:
    if not testo:
        return ""
    i = testo.lower().find("prompt")
    return testo[i:].strip() if i != -1 else testo


def guess_mime(path: str) -> Optional[str]:
    mime, _ = mimetypes.guess_type(path)
    return mime


def make_image_data_url_from_file(path: str) -> Optional[str]:
    if not os.path.isfile(path):
        return None
    mime = guess_mime(path)
    if not mime or mime not in SAFE_IMAGE_MIMES:
        return None
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("ascii")
    return f"data:{mime};base64,{b64}"


def seleziona_immagine_per_azienda(azienda: str):
    filename = IMMAGINI_AZIENDA.get(azienda)
    if not filename:
        return None, None, None, None
    abs_path = os.path.join(UPLOADS_DIR, filename)
    if not os.path.isfile(abs_path):
        log.warning("Reference non trovata per '%s': %s", azienda, abs_path)
        return None, None, None, None
    data_url = make_image_data_url_from_file(abs_path)
    if not data_url:
        log.warning("Reference invalida (mime) per '%s': %s", azienda, abs_path)
        return None, None, None, None
    return f"/uploads/{filename}", filename, data_url, abs_path


def size_profile(formato: str):
    f = (formato or "").lower()
    if f == "storia":  # 9:16
        return {"aspect": "9:16", "target_size": "1080x1920", "api_size": "1024x1536"}
    return {"aspect": "1:1", "target_size": "1024x1024", "api_size": "1024x1024"}


def resize_to_exact(img: "Image.Image", target_w: int, target_h: int) -> "Image.Image":
    tw, th = target_w, target_h
    ta = tw / th
    w, h = img.size
    a = w / h
    if abs(a - ta) < 1e-6:
        base = img
    elif a > ta:
        new_w = int(h * ta)
        x0 = (w - new_w) // 2
        base = img.crop((x0, 0, x0 + new_w, h))
    else:
        new_h = int(w / ta)
        y0 = (h - new_h) // 2
        base = img.crop((0, y0, w, y0 + new_h))
    return base.resize((tw, th), Image.LANCZOS)


def is_org_verification_error(e: Exception) -> bool:
    s = (str(e) or "").lower()
    return (
        ("must be verified" in s)
        or ("verify organization" in s)
        or ("status code: 403" in s)
        or ("error code: 403" in s)
    )

# ────────────────────────────────────────────────────────────────────────────────
# LLM Core: UNA sola conversazione per C1–C3
# ────────────────────────────────────────────────────────────────────────────────

def _single_conversation_responses(
    client,
    azienda: str,
    progetto: str,
    formato: str,
    tipo_slide: str,
    copy_value: str,
    comando1: str,
    comando2: str,
    comando3: str,
    reference_image_data_url: Optional[str],
) -> Dict:
    """
    Una sola chiamata alla Responses API (gpt-5-chat-latest) che restituisce
    un JSON con risposta_comando1/2/3.
    """
    system_msg = {
        "role": "system",
        "content": [
            {
                "type": "input_text",
                "text": (
                    "Sei il motore workflow creativo BD&A.\n"
                    "Riceverai 3 prompt (C1, C2, C3) e devi generare 3 risposte.\n"
                    "Devi restituire SOLO un JSON valido, senza testo extra.\n"
                    "Struttura obbligatoria:\n"
                    "{\n"
                    '  "risposta_comando1": "...",\n'
                    '  "risposta_comando2": "...",\n'
                    '  "risposta_comando3": "..."\n'
                    "}\n"
                    "Non aggiungere spiegazioni fuori dal JSON."
                ),
            }
        ],
    }

    base_info = (
        f"Azienda: {azienda}\n"
        f"Progetto: {progetto}\n"
        f"Formato: {formato}\n"
        f"Tipo slide: {tipo_slide}\n"
        f"Copy: {copy_value}\n"
    )

    prompts_block = (
        "PROMPT_C1:\n"
        f"{comando1}\n\n"
        "PROMPT_C2:\n"
        f"{comando2}\n\n"
        "PROMPT_C3:\n"
        f"{comando3}\n\n"
        "Istruzioni output:\n"
        "- Rispondi a C1 → campo JSON 'risposta_comando1'.\n"
        "- Rispondi a C2 → campo JSON 'risposta_comando2'.\n"
        "- Rispondi a C3 → campo JSON 'risposta_comando3'.\n"
        "Non inserire altri campi.\n"
    )

    user_content = [
        {"type": "input_text", "text": base_info},
        {"type": "input_text", "text": prompts_block},
    ]
    if reference_image_data_url:
        user_content.append({"type": "input_image", "image_url": reference_image_data_url})

    input_payload = [
        system_msg,
        {
            "role": "user",
            "content": user_content,
        },
    ]

    log.info("LLM SINGLE CALL (Responses): model=%s tokens=%s", OPENAI_MODEL_CHAT, MAX_TOKENS_WORKFLOW)
    resp = client.responses.create(
        model=OPENAI_MODEL_CHAT,
        input=input_payload,
        max_output_tokens=MAX_TOKENS_WORKFLOW,
    )

    # openai>=2.8 espone helper output_text
    text = (getattr(resp, "output_text", "") or "").strip()
    if not text:
        # fallback super difensivo
        try:
            first = resp.output[0].content[0]
            text = getattr(first, "text", "") or ""
        except Exception:
            text = ""

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        log.error("JSON decode error Responses. Testo restituito: %s", text)
        raise HTTPException(status_code=502, detail="OpenAI (Responses) ha restituito JSON non valido.")
    return data


def _single_conversation_chat_fallback(
    client,
    azienda: str,
    progetto: str,
    formato: str,
    tipo_slide: str,
    copy_value: str,
    comando1: str,
    comando2: str,
    comando3: str,
) -> Dict:
    """
    Fallback: una sola chiamata Chat Completions (gpt-4o-mini) con lo stesso contratto JSON.
    """
    system = {
        "role": "system",
        "content": (
            "Sei il motore workflow creativo BD&A.\n"
            "Ricevi 3 prompt (C1, C2, C3) e devi rispondere a tutti.\n"
            "Devi restituire SOLO un JSON valido con:\n"
            "{\n"
            '  "risposta_comando1": "...",\n'
            '  "risposta_comando2": "...",\n'
            '  "risposta_comando3": "..."\n'
            "}\n"
            "Nessun testo fuori dal JSON."
        ),
    }

    user = {
        "role": "user",
        "content": (
            f"Azienda: {azienda}\n"
            f"Progetto: {progetto}\n"
            f"Formato: {formato}\n"
            f"Tipo slide: {tipo_slide}\n"
            f"Copy: {copy_value}\n\n"
            "PROMPT_C1:\n"
            f"{comando1}\n\n"
            "PROMPT_C2:\n"
            f"{comando2}\n\n"
            "PROMPT_C3:\n"
            f"{comando3}\n"
        ),
    }

    log.info("LLM SINGLE CALL (ChatCompletions fallback): model=%s tokens=%s", OPENAI_MODEL_FALLBACK, MAX_TOKENS_WORKFLOW)
    resp = client.chat.completions.create(
        model=OPENAI_MODEL_FALLBACK,
        messages=[system, user],
        max_completion_tokens=MAX_TOKENS_WORKFLOW,
    )
    text = (resp.choices[0].message.content or "").strip()
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        log.error("JSON decode error ChatCompletions fallback. Testo restituito: %s", text)
        raise HTTPException(status_code=502, detail="OpenAI (fallback) ha restituito JSON non valido.")
    return data


def llm_single_conversation_with_failover(
    client,
    azienda: str,
    progetto: str,
    formato: str,
    tipo_slide: str,
    copy_value: str,
    comando1: str,
    comando2: str,
    comando3: str,
    reference_image_data_url: Optional[str],
) -> Tuple[Dict, str, str]:
    """
    Ritorna (data_json, backend_usato, model_name) con UNA sola conversazione:
    - primo tentativo: Responses + gpt-5-chat-latest
    - fallback: Chat Completions + gpt-4o-mini
    """
    try:
        data = _single_conversation_responses(
            client,
            azienda,
            progetto,
            formato,
            tipo_slide,
            copy_value,
            comando1,
            comando2,
            comando3,
            reference_image_data_url,
        )
        return data, "responses", OPENAI_MODEL_CHAT
    except Exception as e:
        log.warning("Responses API fallita, fallback Chat Completions. Errore: %s", e)

    try:
        data = _single_conversation_chat_fallback(
            client,
            azienda,
            progetto,
            formato,
            tipo_slide,
            copy_value,
            comando1,
            comando2,
            comando3,
        )
        return data, "chat-completions", OPENAI_MODEL_FALLBACK
    except Exception as e2:
        log.error("Chat Completions fallback fallita: %s", e2)
        raise HTTPException(status_code=502, detail=f"Errore OpenAI: {e2}")

# ────────────────────────────────────────────────────────────────────────────────
# Immagini (opzionale)
# ────────────────────────────────────────────────────────────────────────────────

def generate_image_from_prompt(client, prompt_text: str, formato: str, reference_abs_path: Optional[str]) -> Tuple[str, str, bool]:
    if not IMAGE_API_ENABLED:
        raise HTTPException(status_code=503, detail="API immagini disabilitata (IMAGE_API_ENABLED=false).")

    prof = size_profile(formato)
    api_px = prof["api_size"]
    target_px = prof["target_size"]
    tw, th = [int(x) for x in target_px.split("x")]

    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_name = f"gen-{ts}-{tw}x{th}.png"
    out_path = os.path.join(GENERATED_DIR, out_name)

    def _save_b64_to_png_resized(b64: str):
        raw = base64.b64decode(b64)
        if PIL_AVAILABLE:
            img = Image.open(BytesIO(raw)).convert("RGBA")
            img2 = resize_to_exact(img, tw, th)
            img2.save(out_path, format="PNG")
            return True
        else:
            with open(out_path, "wb") as f:
                f.write(raw)
            return False

    try:
        # 1) tentativo: edit da reference se presente
        if reference_abs_path and os.path.isfile(reference_abs_path):
            try:
                with open(reference_abs_path, "rb") as img_in:
                    log.info("IMAGE CALL (edits): model=%s size=%s", OPENAI_IMAGE_MODEL, api_px)
                    resp = client.images.edits(
                        model=OPENAI_IMAGE_MODEL,
                        image=img_in,
                        prompt=prompt_text,
                        size=api_px,
                    )
                resized = _save_b64_to_png_resized(resp.data[0].b64_json)
                return f"/uploads/generated/{out_name}", out_name, resized
            except Exception as e:
                if is_org_verification_error(e):
                    raise HTTPException(
                        status_code=403,
                        detail=(
                            "OpenAI immagini bloccate: organizzazione non verificata per 'gpt-image-1'. "
                            "Verifica l’organizzazione su platform.openai.com; propagazione fino a 15 minuti."
                        ),
                    )
                log.info("images.edits fallita → images.generate: %s", e)

        # 2) fallback: generate
        log.info("IMAGE CALL (generate): model=%s size=%s", OPENAI_IMAGE_MODEL, api_px)
        resp = client.images.generate(
            model=OPENAI_IMAGE_MODEL,
            prompt=prompt_text,
            size=api_px,
        )
        resized = _save_b64_to_png_resized(resp.data[0].b64_json)
        return f"/uploads/generated/{out_name}", out_name, resized

    except RateLimitError as e:
        log.error("RateLimitError immagini: %s", e)
        raise HTTPException(status_code=429, detail="Rate limit OpenAI (immagini): attendi e riprova.")
    except HTTPException:
        raise
    except Exception as e:
        log.exception("Errore generazione immagine")
        raise HTTPException(status_code=502, detail=f"Errore generazione immagine: {e}")

# ────────────────────────────────────────────────────────────────────────────────
# Core workflow (usa UNA sola conversazione LLM)
# ────────────────────────────────────────────────────────────────────────────────

def workflow_sequenziale(
    azienda: str,
    progetto: str,
    formato: str,
    tipo_slide: str,
    copy_value: str,
    reference_image_data_url: Optional[str],
):
    client = ensure_openai()

    mapping = {
        "azienda": azienda,
        "progetto": progetto,
        "immagine_azienda": IMMAGINI_AZIENDA.get(azienda, ""),
        "tipo_slide": tipo_slide,
        "copy": copy_value,
        "formato": formato,
    }

    # Genero in locale i 3 comandi (templating)
    comando1 = compila_template(template_for("fase1"), mapping)
    t2 = template_for("fase2", tipo_slide)
    if not t2:
        raise HTTPException(status_code=400, detail=f"tipo_slide non valido: {tipo_slide}")
    comando2 = compila_template(t2, mapping)
    t3 = template_for("fase3", azienda)
    if not t3:
        raise HTTPException(status_code=400, detail=f"azienda non valida (fase3): {azienda}")
    comando3 = compila_template(t3, mapping)

    # UNA sola chiamata a OpenAI per C1–C3 (con fallback)
    data, backend, model = llm_single_conversation_with_failover(
        client,
        azienda,
        progetto,
        formato,
        tipo_slide,
        copy_value,
        comando1,
        comando2,
        comando3,
        reference_image_data_url,
    )

    risposta_c1 = data.get("risposta_comando1", "")
    risposta_c2 = data.get("risposta_comando2", "")
    risposta_c3 = data.get("risposta_comando3", "")

    # Prompt filtrato da C3 (dal primo "prompt" in poi)
    prompt3_filtrato = estrai_da_prompt(risposta_c3)

    # C4 (prompt finale per immagine) generato in locale
    comando4 = compila_template(template_for("fase4"), {**mapping, "prompt_generato": prompt3_filtrato})

    return {
        "comando1": comando1,
        "risposta_comando1": risposta_c1,
        "comando2": comando2,
        "risposta_comando2": risposta_c2,
        "comando3": comando3,
        "risposta_comando3": risposta_c3,
        "prompt3_filtrato": prompt3_filtrato,
        "comando4": comando4,
        "llm_backend_used": {
            "C1": {"backend": backend, "model": model, "single_conversation": True},
            "C2": {"backend": backend, "model": model, "single_conversation": True},
            "C3": {"backend": backend, "model": model, "single_conversation": True},
        },
    }


def normalize_response_keys(result: Dict) -> Dict:
    out = dict(result)
    out["risposta1"] = result.get("risposta_comando1", "")
    out["risposta2"] = result.get("risposta_comando2", "")
    out["risposta3"] = result.get("risposta_comando3", "")
    out["comando_1"] = result.get("comando1", "")
    out["comando_2"] = result.get("comando2", "")
    out["comando_3"] = result.get("comando3", "")
    out["comando_4"] = result.get("comando4", "")
    out["prompt_filtrato"] = result.get("prompt3_filtrato", "")
    return out

# ────────────────────────────────────────────────────────────────────────────────
# Endpoints
# ────────────────────────────────────────────────────────────────────────────────

# Frontend: disponibile sia su "/" che su "/frontend1.html"
@app.get("/", response_class=HTMLResponse)
@app.get("/frontend1.html", response_class=HTMLResponse)
def index():
    index_path = os.path.join(BASE_DIR, "frontend1.html")
    if os.path.exists(index_path):
        return FileResponse(index_path, media_type="text/html; charset=utf-8")
    return HTMLResponse("<h1>Frontend non trovato.</h1>", status_code=404)


# Status health-check
@app.get("/status")
def status():
    return {
        "status": "online",
        "openai_ready": bool(OPENAI_API_KEY and OpenAI is not None),
        "pillow_available": PIL_AVAILABLE,
        "image_api_enabled": IMAGE_API_ENABLED,
        "uploads": "/uploads",
        "generated": "/uploads/generated",
        "chat_model": OPENAI_MODEL_CHAT,
        "vision_model": OPENAI_MODEL_VISION,
        "image_model": OPENAI_IMAGE_MODEL,
        "fallback_model": OPENAI_MODEL_FALLBACK,
        "using_failover": True,
        "single_conversation": True,
    }


# OpenAPI YAML endpoint per ChatGPT (in realtà JSON, ma estensione .yaml accettata)
@app.get("/openapi.yaml", include_in_schema=False)
def openapi_yaml():
    return JSONResponse(app.openapi())


@app.post("/api/workflow-text")
async def workflow_text(
    request: Request,
    azienda: str = Form(...),
    progetto: str = Form(...),
    formato: str = Form(...),      # "post"/"carosello" (1:1) oppure "storia" (9:16)
    tipo_slide: str = Form(...),   # "gancio" | "intermedia" | "cta"
    copy_alias: str = Form(None, alias="copy"),
    copy_fallback: str = Form(None, alias="copy_text"),
):
    copy_value = (copy_alias or copy_fallback or "").strip()
    if not copy_value:
        raise HTTPException(status_code=400, detail="Campo 'copy' mancante o vuoto.")

    # Seleziona immagine reference auto (se esiste) → data URL per la vision
    _, _, ref_data_url, _ = seleziona_immagine_per_azienda(azienda)

    result = workflow_sequenziale(
        azienda=azienda,
        progetto=progetto,
        formato=formato,
        tipo_slide=tipo_slide,
        copy_value=copy_value,
        reference_image_data_url=ref_data_url,
    )

    prof = size_profile(formato)
    payload = {
        **normalize_response_keys(result),
        "aspect_ratio": prof["aspect"],
        "size_px": prof["target_size"],
    }
    return JSONResponse(payload)


@app.post("/api/generate-image")
def generate_image(
    azienda: str = Form(...),
    formato: str = Form(...),
    comando4: str = Form(...),
):
    _, _, _, ref_abs = seleziona_immagine_per_azienda(azienda)
    client = ensure_openai()
    url_pubblico, filename, resized = generate_image_from_prompt(
        client=client,
        prompt_text=comando4,
        formato=formato,
        reference_abs_path=ref_abs,
    )
    prof = size_profile(formato)
    payload = {
        "generated_image_url": url_pubblico,
        "generated_image_name": filename,
        "aspect_ratio": prof["aspect"],
        "size_px": prof["target_size"],
    }
    if not PIL_AVAILABLE:
        payload["note"] = "Pillow non installato: immagine salvata alla api_size (nessun resize a target)."
    return JSONResponse(payload)


# Static
if os.path.isdir(STATIC_DIR):
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

app.mount("/uploads", StaticFiles(directory=UPLOADS_DIR), name="uploads")
