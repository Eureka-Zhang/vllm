import json
import time
import urllib.request

from transformers import AutoTokenizer

base = "http://127.0.0.1:8008"
tokenizer = AutoTokenizer.from_pretrained("/home/eureka/qwen38-gguf-lab/tokenizer")
background = tokenizer.encode(
    "这是用于上下文容量测试的背景材料，请忽略重复内容。" * 2000
)
suffix = tokenizer.encode("\n请只回答：测试完成。")
rows = []


def post(path, data):
    try:
        with urllib.request.urlopen(
            urllib.request.Request(
                base + path,
                data=json.dumps(data).encode(),
                headers={"Content-Type": "application/json"},
            ),
            timeout=180,
        ) as r:
            return r.status, json.load(r)
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def rpc(method, args=()):
    status, r = post("/collective_rpc", dict(method=method, args=args))
    assert status == 200, r
    return r["results"][0]


for length in [512, 8192, 15872]:
    prompt = background[: length - len(suffix)] + suffix
    start = time.monotonic()
    status, r = post(
        "/v1/completions",
        dict(model="qwen38-27b-iq3", prompt=prompt, max_tokens=128, temperature=0),
    )
    assert status == 200, r
    assert "测试完成" in r["choices"][0]["text"], r
    stats = rpc("phase_memory_stats")
    row = dict(
        input_tokens=length,
        status=status,
        usage=r["usage"],
        answer=r["choices"][0]["text"],
        seconds=time.monotonic() - start,
        stats=stats,
    )
    rows.append(row)
    print(json.dumps(row, ensure_ascii=False), flush=True)
status, r = post(
    "/v1/completions",
    dict(model="qwen38-27b-iq3", prompt=background[:16300], max_tokens=128),
)
assert status == 400, (status, r)
print("hard limit rejection", status, r, flush=True)
with open("/home/eureka/phase16k-acceptance.json", "w") as result_file:
    result_file.write(json.dumps(rows, ensure_ascii=False, indent=2))
