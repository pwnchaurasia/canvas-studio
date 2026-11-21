import json

import httpx
import pytest

import server
import writer
from conftest import HEADERS

MODELS = "Primary|model-a,Fast Drama|model-b"


class FakeLLM:
    """OpenAI-compatible fake: streams a scripted reply, or returns a fixed completion."""

    def __init__(self):
        self.requests = []
        self.reply_chunks = ["**RIYA:** ", "Baarish ", "ho rahi hai."]
        self.completion = "A woman with no umbrella stands in rain, shot on a camera, warm streetlight, wet asphalt " * 3
        self.status = 200

    def __call__(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        self.requests.append(body)
        if self.status != 200:
            return httpx.Response(self.status, json={"error": {"message": "model overloaded"}})
        if body.get("stream"):
            lines = [{"choices": [{"delta": {"role": "assistant"}}]}]
            lines += [{"choices": [{"delta": {"content": c}}]} for c in self.reply_chunks]
            lines.append({"choices": [], "usage": {"prompt_tokens": 120, "completion_tokens": 9, "cost": 0.0021}})
            sse = "".join(f"data: {json.dumps(x)}\n\n" for x in lines) + "data: [DONE]\n\n"
            return httpx.Response(200, content=sse.encode(), headers={"content-type": "text/event-stream"})
        return httpx.Response(200, json={"choices": [{"message": {"content": self.completion}}],
                                         "usage": {"prompt_tokens": 50, "completion_tokens": 60, "cost": 0.001}})


@pytest.fixture
def llm():
    return FakeLLM()


@pytest.fixture
def wclient(env, renderer, llm, monkeypatch):
    monkeypatch.setenv("LLM_BASE_URL", "https://llm.test/v1")
    monkeypatch.setenv("LLM_API_KEY", "k")
    monkeypatch.setenv("LLM_MODELS", MODELS)
    monkeypatch.setenv("LLM_DEFAULT_MODEL", "model-a")
    monkeypatch.setenv("WRITER_PROMPTS_FILE", str(env / "prompts.local.json"))

    def _make(user, password):
        from fastapi.testclient import TestClient
        app = server.create_app(renderer=renderer, llm_transport=httpx.MockTransport(llm))
        c = TestClient(app)
        assert c.post("/api/login", json={"username": user, "password": password}, headers=HEADERS).status_code == 200
        return c
    return _make


def events(resp):
    return [json.loads(line[6:]) for line in resp.text.splitlines() if line.startswith("data: ")]


def new_thread(c, **kw):
    r = c.post("/api/threads", json={"format_mode": "screenplay", "language": "bilingual", **kw}, headers=HEADERS)
    assert r.status_code == 200, r.text
    return r.json()


# ---------------------------------------------------------------- system prompt


def test_system_prompt_languages_formats_tone_bible():
    base = {"format_mode": "screenplay", "language": "en", "tone_mode": "standard", "story_bible": ""}
    assert "INT./EXT." in writer.build_writer_system_prompt(base)
    assert "Devanagari" in writer.build_writer_system_prompt({**base, "language": "hi"})
    hinglish = writer.build_writer_system_prompt({**base, "language": "hinglish"})
    assert "Roman script" in hinglish and "Do not use Devanagari" in hinglish
    assert "dialogue in natural Hindi or Hinglish" in writer.build_writer_system_prompt({**base, "language": "bilingual"})
    assert "cinematic English" in writer.build_writer_system_prompt(base)
    assert "**HOOK**" in writer.build_writer_system_prompt({**base, "format_mode": "reel"})
    uncut = writer.build_writer_system_prompt({**base, "tone_mode": "uncut"})
    assert "adults (18+)" in uncut and "general audience" not in uncut
    bible = writer.build_writer_system_prompt({**base, "story_bible": "Riya, 27, curly black hair"})
    assert "STORY BIBLE" in bible and "Riya, 27, curly black hair" in bible
    assert "1,500-3,000 words" in writer.build_writer_system_prompt(base, length="long")


def test_local_prompt_override(tmp_path):
    p = tmp_path / "prompts.local.json"
    p.write_text(json.dumps({"uncut": "CUSTOM TONE RULE"}))
    o = writer.load_local_prompts(p)
    s = writer.build_writer_system_prompt({"tone_mode": "uncut", "format_mode": "story", "language": "en"}, o)
    assert "CUSTOM TONE RULE" in s
    assert writer.load_local_prompts(tmp_path / "missing.json") == {}


# ---------------------------------------------------------------- threads


def test_models_endpoint(wclient):
    c = wclient("alice", "pw-a")
    r = c.get("/api/models").json()
    assert r["default"] == "model-a"
    assert [m["id"] for m in r["models"]] == ["model-a", "model-b"]
    assert c.get("/api/me").json()["writer_enabled"] is True


def test_thread_crud(wclient):
    c = wclient("alice", "pw-a")
    t = new_thread(c, story_bible="Riya, 27")
    assert t["model_id"] == "model-a"
    tid = t["id"]
    r = c.patch(f"/api/threads/{tid}", json={"title": "Monsoon", "model_id": "model-b", "story_bible": "Riya, 28"},
                headers=HEADERS)
    assert r.json()["title"] == "Monsoon" and r.json()["model_id"] == "model-b" and r.json()["story_bible"] == "Riya, 28"
    assert [x["id"] for x in c.get("/api/threads").json()["threads"]] == [tid]
    assert c.patch(f"/api/threads/{tid}", json={"model_id": "evil/expensive"}, headers=HEADERS).status_code == 422
    assert c.patch(f"/api/threads/{tid}", json={"language": "fr"}, headers=HEADERS).status_code == 422
    assert c.delete(f"/api/threads/{tid}", headers=HEADERS).status_code == 200
    assert c.get(f"/api/threads/{tid}/messages").status_code == 404


def test_cross_user_isolation(wclient):
    alice, bob = wclient("alice", "pw-a"), wclient("bob", "pw-b")
    tid = new_thread(alice)["id"]
    alice.post(f"/api/threads/{tid}/stream", json={"content": "Scene 1"}, headers=HEADERS)
    assert bob.get("/api/threads").json()["threads"] == []
    for method, path, body in [
        ("get", f"/api/threads/{tid}", None),
        ("get", f"/api/threads/{tid}/messages", None),
        ("get", f"/api/threads/{tid}/export", None),
        ("patch", f"/api/threads/{tid}", {"title": "x"}),
        ("delete", f"/api/threads/{tid}", None),
        ("post", f"/api/threads/{tid}/stream", {"content": "hi"}),
        ("post", f"/api/threads/{tid}/scene-prompt", {"scene_text": "x"}),
        ("post", f"/api/threads/{tid}/render-inline", {"prompt": "x"}),
    ]:
        kwargs = {"headers": HEADERS} if method != "get" else {}
        if body is not None:
            kwargs["json"] = body
        assert getattr(bob, method)(path, **kwargs).status_code == 404, (method, path)
    assert len(alice.get(f"/api/threads/{tid}/messages").json()["messages"]) == 2


# ---------------------------------------------------------------- streaming


def test_stream_saves_messages_usage_and_title(wclient, llm):
    c = wclient("alice", "pw-a")
    tid = new_thread(c, story_bible="Riya, 27, curly hair")["id"]
    r = c.post(f"/api/threads/{tid}/stream", json={"content": "Write the rain scene", "length": "long"}, headers=HEADERS)
    assert r.status_code == 200 and r.headers["content-type"].startswith("text/event-stream")
    ev = events(r)
    assert ev[0]["type"] == "start" and ev[-1]["type"] == "done"
    assert "".join(e["t"] for e in ev if e["type"] == "delta") == "**RIYA:** Baarish ho rahi hai."

    sent = llm.requests[-1]
    assert sent["stream"] is True and sent["model"] == "model-a" and sent["max_tokens"] == 8000
    assert "Riya, 27, curly hair" in sent["messages"][0]["content"]
    assert sent["messages"][-1] == {"role": "user", "content": "Write the rain scene"}

    data = c.get(f"/api/threads/{tid}/messages").json()
    assert data["thread"]["title"] == "Write the rain scene"
    roles = [(m["role"], m["content"]) for m in data["messages"]]
    assert roles == [("user", "Write the rain scene"), ("assistant", "**RIYA:** Baarish ho rahi hai.")]
    assert data["messages"][1]["prompt_tokens"] == 120 and data["messages"][1]["completion_tokens"] == 9
    assert ev[-1]["usage"]["llm_cost_user"] == 0.0021


def test_regenerate_and_continue(wclient, llm):
    c = wclient("alice", "pw-a")
    tid = new_thread(c)["id"]
    c.post(f"/api/threads/{tid}/stream", json={"content": "Scene 1"}, headers=HEADERS)
    llm.reply_chunks = ["Second ", "take."]
    c.post(f"/api/threads/{tid}/stream", json={"regenerate": True}, headers=HEADERS)
    msgs = c.get(f"/api/threads/{tid}/messages").json()["messages"]
    assert [(m["role"], m["content"]) for m in msgs] == [("user", "Scene 1"), ("assistant", "Second take.")]
    assert llm.requests[-1]["messages"][-1] == {"role": "user", "content": "Scene 1"}

    c.post(f"/api/threads/{tid}/stream", json={"content": writer.CONTINUE_TEXT}, headers=HEADERS)
    assert len(c.get(f"/api/threads/{tid}/messages").json()["messages"]) == 4
    assert llm.requests[-1]["messages"][-2] == {"role": "assistant", "content": "Second take."}


def test_stream_upstream_error_reported(wclient, llm):
    c = wclient("alice", "pw-a")
    tid = new_thread(c)["id"]
    llm.status = 503
    ev = events(c.post(f"/api/threads/{tid}/stream", json={"content": "hi"}, headers=HEADERS))
    assert ev[-1]["type"] == "error" and "model overloaded" in ev[-1]["detail"]


def test_uncut_requires_adults(wclient):
    c = wclient("alice", "pw-a")
    assert c.post("/api/threads", json={"tone_mode": "uncut", "story_bible": "Asha, 16 years old"},
                  headers=HEADERS).status_code == 400
    tid = new_thread(c, tone_mode="uncut", story_bible="Asha, 29")["id"]
    assert c.post(f"/api/threads/{tid}/stream", json={"content": "make her a teenager"},
                  headers=HEADERS).status_code == 400
    # standard mode allows family stories with children
    assert c.post("/api/threads", json={"story_bible": "Asha and her two kids"}, headers=HEADERS).status_code == 200


def test_export_markdown(wclient):
    c = wclient("alice", "pw-a")
    tid = new_thread(c, title="Monsoon Nights", story_bible="Riya, 27")["id"]
    c.post(f"/api/threads/{tid}/stream", json={"content": "Scene 1"}, headers=HEADERS)
    r = c.get(f"/api/threads/{tid}/export")
    assert r.status_code == 200 and 'filename="Monsoon-Nights.md"' in r.headers["content-disposition"]
    assert "# Monsoon Nights" in r.text and "Riya, 27" in r.text and "Baarish ho rahi hai." in r.text


# ---------------------------------------------------------------- scene -> image


def test_scene_prompt_cleans_output(wclient, llm):
    c = wclient("alice", "pw-a")
    tid = new_thread(c, story_bible="Riya, 27, curly black hair")["id"]
    r = c.post(f"/api/threads/{tid}/scene-prompt", json={"scene_text": "रिया बारिश में खड़ी है", "style_preset": "custom"},
               headers=HEADERS)
    assert r.status_code == 200, r.text
    out = r.json()
    low = out["visual_prompt"].lower()
    assert "camera" not in low and " no " not in f" {low} "
    assert out["recommended_adapter_scale"] == 0.8
    assert "Riya, 27, curly black hair" in llm.requests[-1]["messages"][1]["content"]
    assert llm.requests[-1]["temperature"] == 0.4


def test_render_inline_saves_thread_and_gallery(wclient, renderer):
    c = wclient("alice", "pw-a")
    tid = new_thread(c)["id"]
    r = c.post(f"/api/threads/{tid}/render-inline",
               json={"prompt": "a woman under a red umbrella", "count": 2, "adapter_scale": 0.8, "seed": 7},
               headers=HEADERS)
    assert r.status_code == 200, r.text
    assert renderer.calls[-1]["count"] == 2 and renderer.calls[-1]["adapter_scale"] == 0.8
    msgs = c.get(f"/api/threads/{tid}/messages").json()["messages"]
    imgs = [m for m in msgs if m["role"] == "image"]
    assert [m["content"]["seed"] for m in imgs] == [7, 8]
    gallery = c.get("/api/gallery").json()["items"]
    assert {g["filename"] for g in gallery} == {m["content"]["filename"] for m in imgs}
    assert c.get(imgs[0]["content"]["url"]).status_code == 200
    assert c.post(f"/api/threads/{tid}/render-inline", json={"prompt": "x", "count": 3}, headers=HEADERS).status_code == 422
