import json
import urllib.request

base = "http://127.0.0.1:8008"


def post(path, d):
    try:
        with urllib.request.urlopen(
            urllib.request.Request(
                base + path,
                data=json.dumps(d).encode(),
                headers={"Content-Type": "application/json"},
            ),
            timeout=90,
        ) as r:
            return r.status, json.load(r)
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def rpc(m, args=()):
    s, r = post("/collective_rpc", dict(method=m, args=args))
    assert s == 200, r
    return r["results"][0]


d = dict(
    model="qwen38-27b-iq3",
    messages=[dict(role="user", content="法国的首都？只回答城市名。")],
    max_tokens=16000,
    temperature=0,
    chat_template_kwargs=dict(enable_thinking=False),
)
rows = []
try:
    rows.append(
        dict(stage="pressure", stats=rpc("phase_test_pressure", [str(1 << 30)]))
    )
    s, r = post("/v1/chat/completions", d)
    assert s == 400, (s, r)
    rows.append(dict(stage="dynamic_reject", status=s, response=r))
    short = dict(d, max_tokens=32)
    s, r = post("/v1/chat/completions", short)
    assert s == 200, r
    assert r["choices"][0]["message"]["content"] == "巴黎", r
    rows.append(dict(stage="small_accepted", stats=rpc("phase_memory_stats")))
finally:
    rows.append(dict(stage="pressure_cleared", stats=rpc("phase_test_pressure", ["0"])))
s, r = post("/v1/chat/completions", d)
assert s == 200, r
assert r["choices"][0]["message"]["content"] == "巴黎", r
rows.append(dict(stage="same_large_budget_accepted", stats=rpc("phase_memory_stats")))
print(json.dumps(rows, ensure_ascii=False, indent=2))
with open("/home/eureka/phase-pressure-acceptance.json", "w") as result_file:
    result_file.write(json.dumps(rows, ensure_ascii=False, indent=2))
