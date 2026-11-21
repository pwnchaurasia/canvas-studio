import json

import httpx
import pytest
from fastapi.testclient import TestClient

import server
import voice
from conftest import HEADERS
from test_writer import FakeLLM

MP3 = b"ID3" + b"\x00" * 2000


class FakeTTS:
    def __init__(self):
        self.calls = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        self.calls.append((request.url.path, body, dict(request.headers)))
        if request.url.path == "/voices/upload":
            return httpx.Response(200, json={"ok": True, "voice_id": body["voice_id"], "seconds": 9.5})
        if request.url.path == "/voices/delete":
            return httpx.Response(200, json={"ok": True, "deleted": True})
        return httpx.Response(200, content=MP3, headers={
            "content-type": "audio/mpeg", "X-Audio-Duration": "4.2", "X-Gen-Seconds": "6.0",
            "X-Cold-Start": "1", "X-Load-Seconds": "30"})


@pytest.fixture
def tts():
    return FakeTTS()


@pytest.fixture
def llm():
    return FakeLLM()


@pytest.fixture
def vclient(env, renderer, llm, tts, monkeypatch):
    monkeypatch.setenv("LLM_BASE_URL", "https://llm.test/v1")
    monkeypatch.setenv("LLM_MODELS", "Primary|model-a")
    monkeypatch.setenv("MODAL_TTS_URL", "https://tts.test")
    monkeypatch.setenv("STUDIO_API_KEY", "studio-key")

    def _make(user, password):
        app = server.create_app(renderer=renderer, llm_transport=httpx.MockTransport(llm),
                                tts_transport=httpx.MockTransport(tts))
        c = TestClient(app)
        assert c.post("/api/login", json={"username": user, "password": password}, headers=HEADERS).status_code == 200
        return c
    return _make


def upload(c, name="Dev", gender="male", lang="en", consent="true", data=b"RIFF" + b"\x01" * 4000, fname="a.wav"):
    return c.post("/api/voices", headers=HEADERS, files={"file": (fname, data, "audio/wav")},
                  data={"display_name": name, "gender": gender, "voice_lang": lang, "preset": "slow_deep", "consent": consent})


# ---------------------------------------------------------------- voice bank


def test_voice_upload_validation(vclient, tts, env):
    c = vclient("alice", "pw-a")
    assert upload(c, consent="false").status_code == 400
    assert upload(c, fname="a.exe").status_code == 422
    assert upload(c, data=b"x" * (5 * 1024 * 1024 + 1)).status_code == 413
    assert upload(c, gender="robot").status_code == 422
    r = upload(c)
    assert r.status_code == 200, r.text
    v = r.json()
    assert v["id"].startswith("m_dev_") and v["default_exaggeration"] == 0.65 and v["default_cfg"] == 0.3
    assert v["consent_confirmed"] == 1 and v["reference_seconds"] == 9.5
    path, body, headers = tts.calls[-1]
    assert path == "/voices/upload" and body["voice_id"] == v["id"] and headers["x-api-key"] == "studio-key"
    assert (env / "voices" / f"{v['id']}.wav").exists()
    listed = c.get("/api/voices").json()
    assert [x["id"] for x in listed["male"]] == [v["id"]] and listed["female"] == []


def test_voice_shared_but_only_owner_deletes(vclient, env):
    alice, bob = vclient("alice", "pw-a"), vclient("bob", "pw-b")
    vid = upload(alice, name="Meera", gender="female", lang="hi").json()["id"]
    assert [x["id"] for x in bob.get("/api/voices").json()["female"]] == [vid]
    assert bob.delete(f"/api/voices/{vid}", headers=HEADERS).status_code == 403
    assert alice.delete(f"/api/voices/{vid}", headers=HEADERS).status_code == 200
    assert not list((env / "voices").glob(f"{vid}.*"))
    assert alice.get("/api/voices").json()["female"] == []


# ---------------------------------------------------------------- single voice


def test_speak_saves_mp3_and_isolates_users(vclient, tts, env):
    alice, bob = vclient("alice", "pw-a"), vclient("bob", "pw-b")
    vid = upload(alice).json()["id"]
    r = alice.post("/api/tts/speak", json={"text": "Hello there.", "voice_id": vid, "language_id": "en"}, headers=HEADERS)
    assert r.status_code == 200, r.text
    out = r.json()
    assert out["duration_sec"] == 4.2 and out["url"].startswith("/api/audio/alice_")
    path, body, _ = tts.calls[-1]
    assert path == "/synthesize" and body["voice_lang"] == "en" and body["exaggeration"] == 0.65
    got = alice.get(out["url"])
    assert got.status_code == 200 and got.content == MP3 and got.headers["content-type"] == "audio/mpeg"
    assert bob.get(out["url"]).status_code == 403
    assert bob.delete(f"/api/audio/{out['id']}", headers=HEADERS).status_code == 404
    for bad in ("..%2Fstudio.db", "alice_1_abc.mp3", "alice_1700000000000_zzzzzz.mp3"):
        assert alice.get(f"/api/audio/{bad}").status_code in (403, 404)
    assert [i["id"] for i in alice.get("/api/audio-history").json()["items"]] == [out["id"]]
    assert bob.get("/api/audio-history").json()["items"] == []
    assert alice.delete(f"/api/audio/{out['id']}", headers=HEADERS).status_code == 200
    assert alice.get(out["url"]).status_code == 404
    # cold start + idle window counted against the shared GPU budget
    assert out["usage"]["gpu_minutes_total"] > 0


def test_hinglish_transliterated_before_tts(vclient, tts, llm):
    c = vclient("alice", "pw-a")
    vid = upload(c).json()["id"]
    llm.completion = "तुम कब आए?"
    r = c.post("/api/tts/speak", json={"text": "Tum kab aaye?", "voice_id": vid, "language_id": "hi"}, headers=HEADERS)
    assert r.status_code == 200 and r.json()["spoken_text"] == "तुम कब आए?"
    assert "Devanagari" in llm.requests[-1]["messages"][0]["content"]
    _, body, _ = tts.calls[-1]
    assert body["text"] == "तुम कब आए?" and body["language_id"] == "hi" and body["voice_lang"] == "en"
    # English output: no transliteration call
    n = len(llm.requests)
    c.post("/api/tts/speak", json={"text": "Hello", "voice_id": vid, "language_id": "en"}, headers=HEADERS)
    assert len(llm.requests) == n


def test_speak_unknown_voice_and_foreign_thread(vclient, llm):
    alice, bob = vclient("alice", "pw-a"), vclient("bob", "pw-b")
    vid = upload(alice).json()["id"]
    assert alice.post("/api/tts/speak", json={"text": "x", "voice_id": "nope"}, headers=HEADERS).status_code == 400
    tid = alice.post("/api/threads", json={}, headers=HEADERS).json()["id"]
    r = bob.post("/api/tts/speak", json={"text": "x", "voice_id": vid, "thread_id": tid}, headers=HEADERS)
    assert r.status_code == 404


# ---------------------------------------------------------------- dialogue


def test_parse_dialogue(vclient, llm):
    c = vclient("alice", "pw-a")
    llm.completion = "Here you go:\n" + json.dumps({"characters": ["Rohan", "Meera"], "turns": [
        {"slot": "narrator", "speaker_name": "Narrator", "text": "Rain hits the window.", "preset": "natural"},
        {"slot": "A", "speaker_name": "Rohan", "text": "**You** came back.", "preset": "slow_deep"},
        {"slot": "b", "speaker_name": "Meera", "text": "I never left.", "preset": "weird"},
        {"slot": "X", "speaker_name": "?", "text": "dropped"},
    ]})
    r = c.post("/api/tts/parse-dialogue", json={"text": "ROHAN: You came back.\nMEERA: I never left."}, headers=HEADERS)
    assert r.status_code == 200, r.text
    t = r.json()["turns"]
    assert [x["slot"] for x in t] == ["narrator", "A", "B"]
    assert t[1]["text"] == "You came back." and t[2]["preset"] == "natural"
    r = c.post("/api/tts/parse-dialogue", json={"text": "x", "include_narrator": False}, headers=HEADERS)
    assert [x["slot"] for x in r.json()["turns"]] == ["A", "B"]
    assert "Omit all narration" in llm.requests[-1]["messages"][0]["content"]


def test_dialogue_render(vclient, tts):
    alice, bob = vclient("alice", "pw-a"), vclient("bob", "pw-b")
    a = upload(alice, name="Rohan").json()["id"]
    b = upload(alice, name="Meera", gender="female", lang="hi").json()["id"]
    turns = [{"slot": "A", "text": "You came back.", "preset": "slow_deep"},
             {"slot": "B", "text": "I never left."},
             {"slot": "narrator", "text": "Silence."}]
    r = alice.post("/api/tts/dialogue", json={"voice_map": {"A": a, "B": b}, "turns": turns}, headers=HEADERS)
    assert r.status_code == 400 and "Narrator" in r.json()["detail"]
    r = alice.post("/api/tts/dialogue", json={"voice_map": {"A": a, "B": b}, "turns": turns[:2]}, headers=HEADERS)
    assert r.status_code == 200, r.text
    path, body, _ = tts.calls[-1]
    assert path == "/synthesize-dialogue" and body["language_id"] == "en"
    assert [(t["speaker"], t["voice_id"], t["voice_lang"]) for t in body["turns"]] == [("A", a, "en"), ("B", b, "hi")]
    assert body["turns"][0]["exaggeration"] == 0.65 and body["turns"][0]["cfg_weight"] == 0.3
    out = r.json()
    assert out["filename"].endswith("_dlg.mp3")
    assert alice.get(out["url"]).status_code == 200 and bob.get(out["url"]).status_code == 403
    clip = alice.get("/api/audio-history").json()["items"][0]
    assert clip["voice_id"] == "dialogue:A+B"


def test_voice_disabled_without_worker(env, renderer):
    app = server.create_app(renderer=renderer)
    c = TestClient(app)
    c.post("/api/login", json={"username": "alice", "password": "pw-a"}, headers=HEADERS)
    assert c.get("/api/me").json()["voice_enabled"] is False
    assert c.post("/api/tts/speak", json={"text": "x", "voice_id": "m_a_1"}, headers=HEADERS).status_code == 503


def test_latin_majority():
    assert voice.latin_majority("Tum kab aaye?") and not voice.latin_majority("तुम कब आए?")
