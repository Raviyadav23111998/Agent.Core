"""Model capability registry — the single source of truth for what the
*currently selected* native model can do, across BOTH directions of multimodal:

  Part 1 — INPUT  (attachments the user sends TO the model): image, pdf, audio, video
  Part 2 — OUTPUT (media the model GENERATES back):          image, audio, video

Why this exists
───────────────
Attachments and media generation are only meaningful if the model the user
picked actually supports them. Rather than scatter `if 'claude' in model`
substring checks across the API and web, every surface asks THIS module:

    caps = capabilities_for(model, provider)
    if not caps["input"]["image"]:
        # WARN the user their selected model can't accept images — do not
        # silently drop the attachment.
    if not caps["output"]["image"]:
        # WARN "the selected model can't generate images" (a provider tool may
        # still cover it — see media_providers).

The web receives this exact object (via the model/agent listing) so the composer
can warn the moment a user attaches an unsupported type, instead of failing deep
in the pipeline.

Design notes
────────────
* Provider-aware: same short name can differ across providers, so we match on
  (model, provider) then fall back to model-family substrings.
* Conservative default: unknown → everything False. A false "unsupported"
  warning is recoverable (pick another model); a false "supported" sends bytes
  to an endpoint that 400s.
* OUTPUT flags describe NATIVE model generation only. Most chat LLMs (Claude
  included) generate no media themselves — Part 2 is delivered by external
  provider tools, and the API merges "native OR provider-configured" before
  deciding whether to warn. This module reports the native half.
"""

from __future__ import annotations

from typing import Dict, Optional


def _caps(image=False, pdf=False, audio_in=False, video_in=False,
          image_gen=False, audio_gen=False, video_gen=False,
          max_image_edge=1568) -> Dict[str, object]:
    return {
        "input":  {"image": image, "pdf": pdf, "audio": audio_in, "video": video_in},
        "output": {"image": image_gen, "audio": audio_gen, "video": video_gen},
        "max_image_edge": max_image_edge,
    }


_DEFAULT = _caps()  # unknown model → nothing supported (safe default)

# Known families. Keys are lowercase substrings matched against the model id.
_FAMILIES: Dict[str, Dict[str, object]] = {
    # Claude — image + PDF input; no native audio/video in; no native media gen.
    "claude":       _caps(image=True, pdf=True),
    # OpenAI GPT-4o/4.1/5 — image input; audio in/out on the -audio variants.
    "gpt-4o":        _caps(image=True, max_image_edge=2048),
    "gpt-4o-audio":  _caps(image=True, audio_in=True, audio_gen=True, max_image_edge=2048),
    "gpt-4.1":       _caps(image=True, max_image_edge=2048),
    "gpt-5":         _caps(image=True, max_image_edge=2048),
    # Native image generators.
    "gpt-image":     _caps(image_gen=True, max_image_edge=2048),
    # Gemini — image/pdf/audio via the OpenAI-compat file/image_url/input_audio
    # parts. video_in stays False until a client emits a video part (no silent
    # drop); flash-image variant also generates images.
    "gemini-2.0-flash-image": _caps(image=True, image_gen=True, max_image_edge=2048),
    "gemini":        _caps(image=True, pdf=True, audio_in=True, video_in=False, max_image_edge=2048),
    # Open vision models.
    "llava":         _caps(image=True),
    "qwen-vl":       _caps(image=True),
    "qwen2-vl":      _caps(image=True),
    "pixtral":       _caps(image=True),
    "deepseek-vl":   _caps(image=True),
    # DeepSeek V4.1 Flash reads images (its /models: input_modalities
    # ["text","image"], 2026-10-10). Only DeepSeek itself calls it this.
    "deepseek-flash": _caps(image=True, max_image_edge=2048),
    "deepseek-v4-flash-vision": _caps(image=True, max_image_edge=2048),
}

# The same id can mean different weights on different providers. DeepSeek
# routes its LEGACY name deepseek-v4-flash to V4.1 Flash, which reads images,
# while a router serving "deepseek/deepseek-v4-flash" serves the text model.
# Checked before the families, for that provider only.
_PROVIDER_FAMILIES: Dict[str, Dict[str, Dict[str, object]]] = {
    "deepseek": {
        "deepseek-flash":    _caps(image=True, max_image_edge=2048),
        "deepseek-v4-flash": _caps(image=True, max_image_edge=2048),
    },
}


def _match(model_lc: str) -> Optional[Dict[str, object]]:
    # Most specific (longest) substring first so "gpt-4o-audio" wins over
    # "gpt-4o" and "deepseek-vl" wins over a text "deepseek".
    for key in sorted(_FAMILIES, key=len, reverse=True):
        if key in model_lc:
            return _FAMILIES[key]
    return None


def capabilities_for(model: Optional[str],
                     provider: Optional[str] = None) -> Dict[str, object]:
    """Capability matrix for a model id (+ optional provider). Never raises —
    an unknown model returns the conservative all-False default so callers warn
    rather than crash."""
    if not model:
        return _clone(_DEFAULT)
    model_lc = str(model).lower().strip()
    for key, caps in sorted(_PROVIDER_FAMILIES.get((provider or "").strip().lower(), {}).items(),
                            key=lambda kv: len(kv[0]), reverse=True):
        if key in model_lc:
            return _clone(caps)
    hit = _match(model_lc)
    return _clone(hit) if hit else _clone(_DEFAULT)


def _clone(c: Dict[str, object]) -> Dict[str, object]:
    return {
        "input":  dict(c["input"]),      # type: ignore[arg-type]
        "output": dict(c["output"]),     # type: ignore[arg-type]
        "max_image_edge": c["max_image_edge"],
    }


# ── Convenience predicates (INPUT side — Part 1) ──────────────────────────────

def supports_image_input(model, provider=None) -> bool:
    return bool(capabilities_for(model, provider)["input"]["image"])   # type: ignore[index]

def supports_pdf_input(model, provider=None) -> bool:
    return bool(capabilities_for(model, provider)["input"]["pdf"])     # type: ignore[index]

def supports_audio_input(model, provider=None) -> bool:
    return bool(capabilities_for(model, provider)["input"]["audio"])   # type: ignore[index]

def supports_video_input(model, provider=None) -> bool:
    return bool(capabilities_for(model, provider)["input"]["video"])   # type: ignore[index]

def accepted_input_kinds(model, provider=None) -> set:
    """The attachment kinds the selected model accepts, e.g. {'image','pdf'}.
    The API uses this to reject/warn on unsupported uploads per selected model."""
    inp = capabilities_for(model, provider)["input"]  # type: ignore[index]
    return {k for k, v in inp.items() if v}


# Backward-compatible aliases used by the vision spike.
def supports_vision(model, provider=None) -> bool:
    return supports_image_input(model, provider)
