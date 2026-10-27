#!/usr/bin/env python3
"""
mock_vllm.py -- a stand-in for a vLLM server, so the Jade tools can be tested
on any laptop without a GPU.

It speaks the two endpoints Jade uses:
  GET  /health            -> 200
  POST /v1/completions    -> streams `max_tokens` chunks as server-sent events,
                             like vLLM with "stream": true, then a usage chunk and [DONE].
Timing: 50 ms before the first token, then 10 ms per token. There is no model:
the point is to exercise the client, proxy and analysis code paths, not to
measure anything.

Usage:  python mock_vllm.py --port 18999
"""
import argparse, asyncio, json

from aiohttp import web


async def health(_):
    return web.Response(text="ok")


async def completions(req):
    body = await req.json()
    n = int(body.get("max_tokens", 16))
    resp = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
    await resp.prepare(req)
    await asyncio.sleep(0.05)                                   # "prefill"
    try:
        for _ in range(n):                                      # "decode", one token per chunk
            await resp.write(f'data: {json.dumps({"choices": [{"text": "x"}]})}\n\n'.encode())
            await asyncio.sleep(0.01)
        await resp.write(f'data: {json.dumps({"choices": [], "usage": {"completion_tokens": n}})}\n\n'.encode())
        await resp.write(b"data: [DONE]\n\n")
    except (ConnectionResetError, asyncio.CancelledError):     # client went away (a cut request)
        pass
    return resp


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--port", type=int, default=18999)
    app = web.Application(client_max_size=64 * 1024 ** 2)
    app.router.add_get("/health", health); app.router.add_post("/v1/completions", completions)
    web.run_app(app, port=ap.parse_args().port, print=None)


if __name__ == "__main__":
    main()
