import concurrent.futures
import json
import threading
import urllib.request

base = "http://127.0.0.1:8008"


def req(path, d):
    return urllib.request.urlopen(
        urllib.request.Request(
            base + path,
            data=json.dumps(d).encode(),
            headers={"Content-Type": "application/json"},
        ),
        timeout=90,
    )


def stats():
    with req("/collective_rpc", dict(method="phase_memory_stats")) as r:
        return json.load(r)["results"][0]


event = threading.Event()
observations = []


def streaming():
    d = dict(
        model="qwen38-27b-iq3",
        messages=[dict(role="user", content="请从1开始依次列出整数。")],
        max_tokens=128,
        ignore_eos=True,
        stream=True,
        chat_template_kwargs=dict(enable_thinking=False),
    )
    count = 0
    with req("/v1/chat/completions", d) as r:
        for line in r:
            if line.startswith(b"data: {"):
                value = json.loads(line[6:])
                delta = value["choices"][0].get("delta", {})
                if delta.get("content"):
                    count += 1
                    if count in [1, 30]:
                        observations.append(stats())
                    if count == 1:
                        event.set()
    return count


with concurrent.futures.ThreadPoolExecutor(2) as executor:
    first = executor.submit(streaming)
    assert event.wait(30)
    d = dict(
        model="qwen38-27b-iq3",
        messages=[dict(role="user", content="17乘以23，只回答数字。")],
        max_tokens=32,
        chat_template_kwargs=dict(enable_thinking=False),
    )

    def second():
        with req("/v1/chat/completions", d) as r:
            return json.load(r)

    later = executor.submit(second)
    chunks = first.result()
    response = later.result()
assert "391" in response["choices"][0]["message"]["content"], response
assert len(observations) == 2, observations
assert observations[0]["blocks"] == observations[1]["blocks"] > 0, observations
assert observations[0]["last_budget"] == observations[1]["last_budget"], observations
print(
    json.dumps(
        dict(
            status="passed",
            stream_chunks=chunks,
            active_snapshots=observations,
            queued_response=response["choices"][0]["message"]["content"],
            idle=stats(),
        ),
        ensure_ascii=False,
        indent=2,
    )
)
