# llama.cpp LoRA shim

One `llama-server`, several model ids. A small OpenAI-compatible proxy that gives each LoRA adapter its own model id, the way
vLLM's `--lora-modules` does, on one engine with shared weights and slots.

```text
client  -->  lora-shim.py :8180  -->  llama-server :8190 (--lora a.gguf --lora b.gguf --lora-init-without-apply)
             /v1/models: base, base-a, base-b
             model "base-a"  ->  model "base" + {"lora": [{"id": 0, "scale": 1.0}]}
```

- Pure Python standard library, one file, no dependencies. Python 3.9+. Windows, macOS, Linux.
- The extra memory per adapter is the adapter file itself (a few hundred MB), not a second copy of the model.
- Everything that is not a chat or completion request passes through unchanged: streaming, health, metrics, embeddings,
  `/props`, `/slots`.

## Why

`llama-server` can load adapters at start and switch them per request with the `lora` body field, but it exposes one model id.
Clients such as Open WebUI, Hermes and OpenClaw pick models by id, and a persona that needs the adapter should not have to know
about a body field. Running a second llama-server per adapter costs a full copy of the weights and its own KV cache. This shim
gives each adapter a name at the cost of nothing.

## Quick start

1. Start llama-server with every adapter preloaded at scale 0, on a loopback port:

   ```sh
   llama-server -m base.gguf --alias base --host 127.0.0.1 --port 8190 \
     --lora a.gguf --lora b.gguf --lora-init-without-apply ...
   ```

   `--lora-init-without-apply` is required. Without it llama-server applies every adapter at scale 1 to every request.

2. Start the shim on the public port, naming the adapters in the same order as the `--lora` flags:

   ```sh
   python3 lora-shim.py --listen 8180 --listen-host 0.0.0.0 --upstream 8190 --base base --lora base-a --lora base-b
   ```

3. Point clients at `http://<host>:8180/v1`. `/v1/models` lists `base`, `base-a` and `base-b`.

Add `-- llama-server ...` after the shim's flags to have the shim start llama-server as a child and exit when it exits.

## Flags

| Flag | Meaning |
| --- | --- |
| `--listen PORT`, `--listen-host H` | Where clients connect. Default host `0.0.0.0`. |
| `--upstream PORT` | llama-server's port on `127.0.0.1`. |
| `--base ALIAS` | llama-server's `--alias`. Requests for this id, or any unknown id, go through with every adapter at scale 0. |
| `--lora ID` | One per adapter, in the order of llama-server's `--lora` flags. The first is adapter 0. |
| `--lora-slot N` | Pin every adapter request to slot N when the request carries no `id_slot` of its own. See below. |
| `-- CMD...` | Run llama-server as a child. |

`SHIM_DUMP_DIR=/some/dir` in the environment saves every rewritten chat request body to that directory, for prompt diffing when
a prefix cache is not being reused. Off unless set.

## Slots and the adapter rule

llama-server keeps one prompt cache per slot, and **clears a slot's cache whenever the adapter set on that slot differs from the
previous request's** (`launch_slot_with_task` → `lora_should_clear_cache`). On a multi-slot server with unpinned requests, a base
chat and an adapter chat that alternate on the same slot wipe each other's cache every time.

`--lora-slot N` keeps the adapter traffic on one slot, so the base keeps the others and nobody's prefix is cleared by an
adapter switch. With `-np 3` and one adapter, `--lora-slot 2` leaves slots 0 and 1 for the base. On a one-slot server leave it
unset. The pin is only applied when the request has no `id_slot`; a client or an upstream proxy that sets one wins.

## Chaining with llama.cpp-ssd-cache-shim

[llama.cpp-ssd-cache-shim](https://github.com/beaudamore/llama.cpp-ssd-cache-shim) keeps each conversation's slot state on SSD
and restores it when the conversation returns. Put that shim on the public port and this one on a loopback port behind it:

```text
clients --> ssd_cache_shim.py :8180 --> lora-shim.py :8191 (127.0.0.1) --> llama-server :8190 (127.0.0.1)
```

The cache shim sees the original model id, so adapter chats get their own key, and it pins every request with `id_slot`, which
this shim forwards untouched. Give the cache shim a `MODEL_SLOTS` entry for each adapter id (`MODEL_SLOTS=base-a:2`), the same
lane `--lora-slot` would have used, so the adapter rule above still holds. What this shim guarantees for that chain:

- every request header is forwarded;
- the body is forwarded with only `model` and `lora` rewritten, so an `id_slot` set upstream survives;
- `--lora-slot` yields to an existing `id_slot`;
- `/props`, `/slots` and `/slots/{id}?action=save|restore` are proxied unchanged.

Listen on `127.0.0.1` in that setup. A request that reaches this shim without going through the cache shim lands on a slot the
cache shim does not know about.

## How the rewrite works

- A request whose `model` is an adapter id is sent upstream with `model` set to the base and
  `"lora": [{"id": N, "scale": 1.0}]`. The response's `model` field is rewritten back to the requested id: in full for
  non-streaming replies, chunk by chunk for `text/event-stream`.
- A request for the base, or for an unknown id, is sent with every adapter listed at scale 0. An empty `lora` list would mean
  "server defaults", and the defaults keep each `--lora` at scale 1 even with `--lora-init-without-apply`.
- `/v1/models` is fetched from llama-server and one entry per adapter id is appended, copied from the base entry.

## Limits

- One adapter per request. Mixing adapters, or scales other than 1, needs the raw `lora` field, which still works for a request
  sent as the base model.
- The streaming rewrite is a byte replacement of `"model":"<base>"` inside chunks. It has never missed in practice, but it is a
  string substitution, not a parse.
- Requests with different adapter sets are not batched together by llama-server, which is another reason for `--lora-slot`.

## Requirements

- A llama-server that accepts the per-request `lora` field and `--lora-init-without-apply` (both from 2024 onward).
- Python 3.9 or newer.
