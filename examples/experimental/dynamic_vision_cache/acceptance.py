import base64
import io
import json
import time
import urllib.request

from PIL import Image

url = "http://127.0.0.1:8008"


def post(path, d):
    return urllib.request.urlopen(
        urllib.request.Request(
            url + path,
            data=json.dumps(d).encode(),
            headers={"Content-Type": "application/json"},
        ),
        timeout=90,
    )


def stats():
    return json.load(post("/collective_rpc", {"method": "phase_memory_stats"}))[
        "results"
    ][0]


rows = []
for color in [None, "blue", None, "red"]:
    content = "17乘以23等于多少？只回答数字。"
    if color:
        b = io.BytesIO()
        Image.new("RGB", (224, 224), color).save(b, format="PNG")
        content = [
            dict(type="text", text="图片主要是什么颜色？只回答颜色。"),
            dict(
                type="image_url",
                image_url=dict(
                    url="data:image/png;base64,"
                    + base64.b64encode(b.getvalue()).decode()
                ),
            ),
        ]
    d = dict(
        model="qwen38-27b-iq3",
        messages=[dict(role="user", content=content)],
        temperature=0,
        max_tokens=32,
        chat_template_kwargs=dict(enable_thinking=False),
    )
    r = json.load(post("/v1/chat/completions", d))
    ans = r["choices"][0]["message"]["content"]
    assert (
        "391" in ans if color is None else ("蓝" if color == "blue" else "红") in ans
    ), r
    deadline = time.monotonic() + 10
    while True:
        s = stats()
        if s["cache_bytes"] == 0:
            break
        if time.monotonic() > deadline:
            raise AssertionError(s)
        time.sleep(0.1)
    rows.append(dict(color=color, answer=ans, stats=s))
    print(json.dumps(rows[-1], ensure_ascii=False), flush=True)
d = dict(
    model="qwen38-27b-iq3",
    messages=[dict(role="user", content="请详细写一篇关于人工智能历史的长文。")],
    max_tokens=1000,
    stream=True,
)
with post("/v1/chat/completions", d) as response:
    for line in response:
        if b"content" in line:
            break
# Disconnect during generation, then wait for scheduler cleanup.
deadline = time.monotonic() + 15
while True:
    s = stats()
    if s["cache_bytes"] == 0:
        break
    if time.monotonic() > deadline:
        raise AssertionError(s)
    time.sleep(0.1)
print(json.dumps(dict(cancel_cleanup=s)), flush=True)
with open("/home/eureka/dynamic-acceptance.json", "w") as result_file:
    result_file.write(json.dumps(rows, ensure_ascii=False, indent=2))
