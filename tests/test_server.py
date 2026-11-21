from conftest import HEADERS


def gen(c, **kw):
    body = {"prompt": "portrait of a woman in a linen shirt", "count": 2, **kw}
    return c.post("/api/generate", json=body, headers=HEADERS)


# ---------------------------------------------------------------- auth

def test_api_requires_login(make_client):
    c = make_client()
    assert c.get("/api/me").status_code == 401
    assert c.get("/api/gallery").status_code == 401
    assert gen(c).status_code == 401


def test_bad_password_rejected(make_client):
    c = make_client()
    r = c.post("/api/login", json={"username": "alice", "password": "nope"}, headers=HEADERS)
    assert r.status_code == 401
    r = c.post("/api/login", json={"username": "mallory", "password": "x"}, headers=HEADERS)
    assert r.status_code == 401


def test_login_sets_httponly_cookie(make_client):
    c = make_client()
    r = c.post("/api/login", json={"username": "alice", "password": "pw-a"}, headers=HEADERS)
    cookie = r.headers["set-cookie"].lower()
    assert "httponly" in cookie and "samesite=lax" in cookie and "max-age=2592000" in cookie
    assert c.get("/api/me").json()["user"] == "alice"


def test_forged_cookie_rejected(make_client):
    c = make_client()
    c.cookies.set("cs_session", "alice.forged.sig")
    assert c.get("/api/me").status_code == 401


def test_login_rate_limit(make_client):
    c = make_client()
    for _ in range(5):
        c.post("/api/login", json={"username": "alice", "password": "bad"}, headers=HEADERS)
    r = c.post("/api/login", json={"username": "alice", "password": "pw-a"}, headers=HEADERS)
    assert r.status_code == 429


def test_mutations_require_csrf_header(make_client):
    c = make_client("alice", "pw-a")
    r = c.post("/api/generate", json={"prompt": "x"})
    assert r.status_code == 403


# ---------------------------------------------------------------- gallery isolation

def test_user_isolation(make_client):
    alice = make_client("alice", "pw-a")
    bob = make_client("bob", "pw-b")
    r = gen(alice)
    assert r.status_code == 200, r.text
    name = r.json()["images"][0]["filename"]
    assert name.startswith("alice_")

    assert alice.get(f"/api/image/{name}").status_code == 200
    assert bob.get(f"/api/image/{name}").status_code == 404
    assert bob.delete(f"/api/image/{name}", headers=HEADERS).status_code == 404
    assert bob.get("/api/gallery").json()["items"] == []
    assert len(alice.get("/api/gallery").json()["items"]) == 2


def test_spoofed_prefix_rejected(make_client, env):
    bob = make_client("bob", "pw-b")
    # a file named like bob's on disk, but not recorded for bob in the DB
    (env / "gallery" / "bob_1700000000000_1.webp").write_bytes(b"x")
    assert bob.get("/api/image/bob_1700000000000_1.webp").status_code == 404


def test_path_traversal_rejected(make_client, env):
    (env / "secret.txt").write_text("top secret")
    alice = make_client("alice", "pw-a")
    for bad in ("..%2Fsecret.txt", "..%2F..%2Fetc%2Fpasswd", "alice_1_1.webp%2F..%2F..%2Fsecret.txt",
                "%2E%2E%2Fstudio.db", "alice_1700000000000_1.webp%00.txt"):
        r = alice.get(f"/api/image/{bad}")
        assert r.status_code == 404, bad
        assert b"top secret" not in r.content


def test_delete_own_image(make_client, env):
    alice = make_client("alice", "pw-a")
    name = gen(alice, count=1).json()["images"][0]["filename"]
    assert (env / "gallery" / name).exists()
    assert alice.delete(f"/api/image/{name}", headers=HEADERS).status_code == 200
    assert not (env / "gallery" / name).exists()
    assert alice.get("/api/gallery").json()["items"] == []


# ---------------------------------------------------------------- schema validation

def test_schema_limits(make_client):
    c = make_client("alice", "pw-a")
    assert gen(c, count=5).status_code == 422
    assert gen(c, count=0).status_code == 422
    assert gen(c, aspect="4:3").status_code == 422
    assert gen(c, steps=13).status_code == 422
    assert gen(c, adapter_scale=1.5).status_code == 422
    assert gen(c, prompt="").status_code == 422


def test_generate_passes_params_and_records_usage(make_client, renderer):
    c = make_client("alice", "pw-a")
    r = gen(c, count=3, seed=42, aspect="1:1", steps=6, adapter_scale=0.8)
    assert r.status_code == 200
    body = r.json()
    assert [i["seed"] for i in body["images"]] == [42, 43, 44]
    assert renderer.calls[-1] == {"prompt": "portrait of a woman in a linen shirt", "aspect": "1:1",
                                  "count": 3, "steps": 6, "seed": 42, "adapter_scale": 0.8}
    # 4s generation + 25s idle window + 20s cold load
    assert body["estimated_gpu_sec"] == 49.0
    assert body["usage"]["gpu_minutes_user"] == round(49 / 60, 1)


def test_monthly_cap_blocks(make_client, monkeypatch):
    monkeypatch.setenv("MONTHLY_GPU_MINUTES_CAP", "0.5")
    c = make_client("alice", "pw-a")
    assert gen(c).status_code == 200  # 49s used
    assert gen(c).status_code == 429


def test_content_guard(make_client, renderer):
    c = make_client("alice", "pw-a")
    assert gen(c, prompt="a 16 year old at the beach").status_code == 400
    assert gen(c, prompt="teenage girl portrait").status_code == 400
    assert renderer.calls == []


# ---------------------------------------------------------------- hygiene

def test_docs_disabled_and_robots(make_client):
    c = make_client()
    for p in ("/docs", "/redoc", "/openapi.json"):
        assert c.get(p).status_code == 404
    r = c.get("/robots.txt")
    assert r.text == "User-agent: *\nDisallow: /\n"
    assert r.headers["x-robots-tag"] == "noindex, nofollow"
    assert 'name="robots" content="noindex, nofollow"' in c.get("/").text


def test_chat_disabled_without_llm(make_client):
    c = make_client("alice", "pw-a")
    assert c.get("/api/me").json()["chat_enabled"] is False
    r = c.post("/api/chat", json={"messages": [{"role": "user", "content": "hi"}]}, headers=HEADERS)
    assert r.status_code == 503
