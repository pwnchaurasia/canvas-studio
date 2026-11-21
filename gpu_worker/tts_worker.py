"""Canvas Studio voice worker: Chatterbox Multilingual TTS on a scale-to-zero Modal T4.

Deploy:  modal deploy gpu_worker/tts_worker.py
Routes (one URL, edge proxy auth + X-API-Key):
  POST /voices/upload        {voice_id, audio_b64, filename}
  POST /voices/delete        {voice_id}
  POST /synthesize           {text, voice_id, language_id, voice_lang, exaggeration, cfg_weight}
  POST /synthesize-dialogue  {language_id, turns: [{speaker, voice_id, voice_lang, text, exaggeration, cfg_weight}]}
"""

import base64
import hmac
import os
import re
import subprocess
import tempfile
import time

import modal

APP_NAME = "canvas-studio-tts"
CACHE_DIR = "/cache"
VOICE_DIR = f"{CACHE_DIR}/voices"
SECRET_NAME = "canvas-studio-secrets"

MAX_REF_SECONDS = 15
CHUNK_CHARS = 280
MAX_TURNS = 30
MAX_TEXT_CHARS = 8000
TARGET_DBFS = -16.0
GAP_SENTENCE_MS, GAP_SAME_SPEAKER_MS, GAP_SPEAKER_CHANGE_MS = 250, 200, 400
VOICE_ID_RE = re.compile(r"^[a-z0-9_-]{1,48}$")
LANGS = ("en", "hi")

image = (
    modal.Image.debian_slim(python_version="3.11")  # chatterbox-tts pins deps tested on 3.11
    .apt_install("ffmpeg", "git")
    .uv_pip_install("chatterbox-tts==0.1.7", "fastapi[standard]", "setuptools", "soundfile")
    .env({"HF_HOME": CACHE_DIR, "HF_HUB_ENABLE_HF_TRANSFER": "0"})
)

tts_vol = modal.Volume.from_name("canvas-tts-cache", create_if_missing=True)
secrets = [modal.Secret.from_name(SECRET_NAME)]
app = modal.App(APP_NAME)


# ---------------------------------------------------------------- text helpers (pure, unit-testable)


def clean_for_speech(text: str) -> str:
    """Remove markdown and stage directions that should not be read aloud."""
    t = re.sub(r"```.*?```", " ", text, flags=re.S)
    t = re.sub(r"^\s{0,3}#{1,6}\s*", "", t, flags=re.M)          # headings
    t = re.sub(r"^\s*([-*•]|\d+[.)])\s+", "", t, flags=re.M)     # list markers
    t = re.sub(r"^\s*>\s?", "", t, flags=re.M)                    # quotes
    t = re.sub(r"^\s*(-{3,}|\*{3,}|_{3,})\s*$", " ", t, flags=re.M)
    t = re.sub(r"\[[^\]]{0,200}\]", " ", t)                       # [stage cues]
    t = re.sub(r"[*_`~]+", "", t)                                 # emphasis marks
    t = re.sub(r"^\s*[A-Z][A-Z .'-]{1,30}(\([^)]*\))?\s*:\s*", "", t, flags=re.M)  # "RIYA:" speaker labels
    t = re.sub(r"[ \t]+", " ", t)
    t = re.sub(r"\s*\n\s*", "\n", t)
    return t.strip()


def chunk_text(text: str, limit: int = CHUNK_CHARS) -> list[str]:
    """Split on sentence boundaries (incl. Hindi danda) into chunks of at most `limit` chars."""
    sentences = [s.strip() for s in re.split(r"(?<=[.!?।॥…])\s+|\n+", text) if s and s.strip()]
    pieces: list[str] = []
    for s in sentences:
        while len(s) > limit:  # very long sentence: break at the last comma/space before the limit
            cut = max(s.rfind(",", 0, limit), s.rfind("،", 0, limit), s.rfind(" ", 0, limit))
            cut = cut if cut > limit // 3 else limit
            pieces.append(s[: cut + 1].strip())
            s = s[cut + 1 :].strip()
        if s:
            pieces.append(s)
    chunks: list[str] = []
    for p in pieces:
        if chunks and len(chunks[-1]) + 1 + len(p) <= limit:
            chunks[-1] = f"{chunks[-1]} {p}"
        else:
            chunks.append(p)
    return chunks


# ---------------------------------------------------------------- worker


@app.cls(
    image=image,
    gpu="T4",
    volumes={CACHE_DIR: tts_vol},
    secrets=secrets,
    scaledown_window=20,
    min_containers=0,
    buffer_containers=0,
    max_containers=1,
    timeout=900,
)
class TTSWorker:
    @modal.enter()
    def load(self) -> None:
        from chatterbox.mtl_tts import ChatterboxMultilingualTTS

        t0 = time.perf_counter()
        os.makedirs(VOICE_DIR, exist_ok=True)
        self.model = ChatterboxMultilingualTTS.from_pretrained(device="cuda")
        self.sr = self.model.sr
        self._conds: dict[str, tuple[float, object]] = {}  # voice_id -> (file mtime, conditionals)
        try:
            tts_vol.commit()  # persist weights downloaded on a first-ever run
        except Exception:
            pass
        self.load_s = round(time.perf_counter() - t0, 2)
        self.cold = True
        print(f"Chatterbox ready in {self.load_s}s")

    # ---- audio helpers

    def _voice_path(self, voice_id: str) -> str:
        from fastapi import HTTPException

        if not VOICE_ID_RE.fullmatch(voice_id or ""):
            raise HTTPException(status_code=422, detail="invalid voice_id")
        path = f"{VOICE_DIR}/{voice_id}.wav"
        if not os.path.exists(path):
            tts_vol.reload()
        if not os.path.exists(path):
            raise HTTPException(status_code=404, detail=f"voice '{voice_id}' not found on worker")
        return path

    def _use_voice(self, voice_id: str, exaggeration: float) -> None:
        path = self._voice_path(voice_id)
        mtime = os.path.getmtime(path)
        cached = self._conds.get(voice_id)
        if not cached or cached[0] != mtime:
            self.model.prepare_conditionals(path, exaggeration=exaggeration)
            self._conds[voice_id] = (mtime, self.model.conds)
        self.model.conds = self._conds[voice_id][1]

    def _speak(self, text: str, language_id: str, exaggeration: float, cfg_weight: float):
        import numpy as np

        wav = self.model.generate(text, language_id=language_id, exaggeration=exaggeration, cfg_weight=cfg_weight)
        return wav.squeeze(0).cpu().numpy().astype(np.float32)

    @staticmethod
    def _normalize(x, target_dbfs: float = TARGET_DBFS):
        import numpy as np

        rms = float(np.sqrt(np.mean(np.square(x)))) if x.size else 0.0
        if rms < 1e-5:
            return x
        gain = 10 ** ((target_dbfs - 20 * np.log10(rms)) / 20)
        y = x * gain
        peak = float(np.max(np.abs(y)))
        return y * (0.98 / peak) if peak > 0.98 else y

    def _silence(self, ms: int):
        import numpy as np

        return np.zeros(int(self.sr * ms / 1000), dtype=np.float32)

    def _mp3(self, x, frames_only: bool = False) -> bytes:
        """128 kbps MP3. frames_only drops the ID3/Xing headers so parts can be byte-concatenated."""
        import numpy as np

        extra = ["-write_xing", "0", "-id3v2_version", "0", "-write_id3v1", "0"] if frames_only else []
        proc = subprocess.run(
            ["ffmpeg", "-hide_banner", "-loglevel", "error", "-f", "f32le", "-ar", str(self.sr), "-ac", "1",
             "-i", "pipe:0", "-codec:a", "libmp3lame", "-b:a", "128k", *extra, "-f", "mp3", "pipe:1"],
            input=np.ascontiguousarray(x, dtype=np.float32).tobytes(), capture_output=True, check=True,
        )
        return proc.stdout

    def _response(self, audio, t0: float, frames_only: bool = False, lead_silence_ms: int = 0):
        import numpy as np
        from fastapi import Response

        if lead_silence_ms:
            audio = np.concatenate([self._silence(lead_silence_ms), audio])
        duration = len(audio) / self.sr
        gen_s = round(time.perf_counter() - t0, 2)
        cold, self.cold = self.cold, False
        headers = {
            "X-Audio-Duration": str(round(duration, 2)),
            "X-Gen-Seconds": str(gen_s),
            "X-Cold-Start": "1" if cold else "0",
            "X-Load-Seconds": str(self.load_s if cold else 0.0),
        }
        print(f"Synthesized {duration:.1f}s of audio in {gen_s}s (cold={cold})")
        return Response(content=self._mp3(audio, frames_only), media_type="audio/mpeg", headers=headers)

    # ---- web app

    @modal.asgi_app(requires_proxy_auth=True)
    def web(self):
        import numpy as np
        from fastapi import FastAPI, Header, HTTPException
        from pydantic import BaseModel, Field

        api = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)

        def check_key(key: str | None) -> None:
            expected = os.environ.get("STUDIO_API_KEY", "")
            if not expected or not key or not hmac.compare_digest(key, expected):
                raise HTTPException(status_code=401, detail="unauthorized")

        class Upload(BaseModel):
            voice_id: str = Field(pattern=VOICE_ID_RE.pattern)
            audio_b64: str = Field(max_length=8_000_000)
            filename: str = Field(default="clip", max_length=200)

        class VoiceRef(BaseModel):
            voice_id: str = Field(pattern=VOICE_ID_RE.pattern)

        class Synth(BaseModel):
            text: str = Field(min_length=1, max_length=MAX_TEXT_CHARS)
            voice_id: str = Field(pattern=VOICE_ID_RE.pattern)
            language_id: str = Field(default="en", pattern="^(en|hi)$")
            voice_lang: str = Field(default="en", pattern="^(en|hi)$")
            exaggeration: float = Field(default=0.5, ge=0.0, le=2.0)
            cfg_weight: float = Field(default=0.5, ge=0.0, le=1.0)
            frames_only: bool = False
            lead_silence_ms: int = Field(default=0, ge=0, le=2000)

        class Turn(BaseModel):
            speaker: str = Field(default="A", max_length=32)
            voice_id: str = Field(pattern=VOICE_ID_RE.pattern)
            voice_lang: str = Field(default="en", pattern="^(en|hi)$")
            text: str = Field(min_length=1, max_length=2000)
            exaggeration: float = Field(default=0.5, ge=0.0, le=2.0)
            cfg_weight: float = Field(default=0.5, ge=0.0, le=1.0)

        class Dialogue(BaseModel):
            language_id: str = Field(default="en", pattern="^(en|hi)$")
            turns: list[Turn] = Field(min_length=1, max_length=MAX_TURNS)
            frames_only: bool = False
            lead_silence_ms: int = Field(default=0, ge=0, le=2000)

        @api.post("/voices/upload")
        def upload(req: Upload, x_api_key: str | None = Header(default=None)):
            check_key(x_api_key)
            raw = base64.b64decode(req.audio_b64)
            os.makedirs(VOICE_DIR, exist_ok=True)
            out = f"{VOICE_DIR}/{req.voice_id}.wav"
            with tempfile.NamedTemporaryFile(suffix=os.path.splitext(req.filename)[1] or ".bin") as src:
                src.write(raw)
                src.flush()
                proc = subprocess.run(
                    ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", src.name, "-t", str(MAX_REF_SECONDS),
                     "-ac", "1", "-ar", "24000", "-af", "loudnorm=I=-18:TP=-2", out],
                    capture_output=True,
                )
            if proc.returncode != 0 or not os.path.exists(out):
                raise HTTPException(status_code=422, detail="could not decode audio clip")
            self._conds.pop(req.voice_id, None)
            tts_vol.commit()
            dur = float(subprocess.run(
                ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", out],
                capture_output=True, text=True,
            ).stdout.strip() or 0)
            return {"ok": True, "voice_id": req.voice_id, "seconds": round(dur, 2)}

        @api.post("/voices/delete")
        def delete(req: VoiceRef, x_api_key: str | None = Header(default=None)):
            check_key(x_api_key)
            path = f"{VOICE_DIR}/{req.voice_id}.wav"
            existed = os.path.exists(path)
            if existed:
                os.remove(path)
                tts_vol.commit()
            self._conds.pop(req.voice_id, None)
            return {"ok": True, "deleted": existed}

        @api.post("/synthesize")
        def synthesize(req: Synth, x_api_key: str | None = Header(default=None)):
            check_key(x_api_key)
            t0 = time.perf_counter()
            cfg = 0.0 if req.voice_lang != req.language_id else req.cfg_weight  # limits accent carry-over
            self._use_voice(req.voice_id, req.exaggeration)
            chunks = chunk_text(clean_for_speech(req.text))
            if not chunks:
                raise HTTPException(status_code=422, detail="nothing to speak after cleanup")
            parts = []
            for i, chunk in enumerate(chunks):
                if i:
                    parts.append(self._silence(GAP_SENTENCE_MS))
                parts.append(self._speak(chunk, req.language_id, req.exaggeration, cfg))
            return self._response(self._normalize(np.concatenate(parts)), t0, req.frames_only, req.lead_silence_ms)

        @api.post("/synthesize-dialogue")
        def synthesize_dialogue(req: Dialogue, x_api_key: str | None = Header(default=None)):
            check_key(x_api_key)
            t0 = time.perf_counter()
            for vid in {t.voice_id for t in req.turns}:
                self._voice_path(vid)  # fail fast (one reload) before spending GPU time
            parts, prev = [], None
            for turn in req.turns:
                chunks = chunk_text(clean_for_speech(turn.text))
                if not chunks:
                    continue
                cfg = 0.0 if turn.voice_lang != req.language_id else turn.cfg_weight
                self._use_voice(turn.voice_id, turn.exaggeration)
                seg = []
                for i, chunk in enumerate(chunks):
                    if i:
                        seg.append(self._silence(GAP_SAME_SPEAKER_MS))
                    seg.append(self._speak(chunk, req.language_id, turn.exaggeration, cfg))
                if prev is not None:
                    parts.append(self._silence(GAP_SPEAKER_CHANGE_MS if turn.speaker != prev else GAP_SAME_SPEAKER_MS))
                parts.append(self._normalize(np.concatenate(seg)))  # per-turn loudness match
                prev = turn.speaker
            if not parts:
                raise HTTPException(status_code=422, detail="nothing to speak after cleanup")
            return self._response(np.concatenate(parts), t0, req.frames_only, req.lead_silence_ms)

        return api
