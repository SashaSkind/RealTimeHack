# Using VAST Data at the Builders Challenge (SF, 2026-10-02)

Today's event is the **VAST Builders Challenge: Real-Time Video Agents Hack** (AWS Builder Loft, build 10:30, demos 17:30). Starter repo: <https://github.com/vast-data/vast-builders-challenge>.

## What the event gives each team

- **A VM with Cursor.** It holds the starter repo and the team config `/config/<team>.config`, which contains every credential below.
- **A video pipeline already running on VAST:** S3 → DataEngine (Segmenter → YOLO11 Detector → Cosmos3-Reason Reasoner → Cosmos Embed1 Embedder → VastDB writer). The result is a searchable VastDB index of a pre-ingested corpus: traffic, driving, street cams, a warehouse and indoor scenes.
- **Shared GPU models on CoreWeave:**

  | Model | URL | Port | Interface |
  |---|---|---|---|
  | `nvidia/cosmos3-reason` | `$COSMOS3_REASON_URL` | 8001 | OpenAI-compatible chat, image/video input |
  | YOLO11 | `$YOLO_URL` | 8002 | |
  | `nvidia/cosmos-embed1` | `$COSMOS_EMBED1_URL` | 8003 | 256-dim text and video embeddings |
  | Canary-1B | `$CANARY_1B_URL` | 8004 | speech to text |

  All of them take `Authorization: Bearer $GPU_BEARER_TOKEN`.
- **Weights & Biases serverless LLM inference:** the `WANDB_*` keys in the VM's environment.

**Checked from this Mac (2026-10-02):** the Cosmos3-Reason host answers in 0.12 s (401 without a token), so the laptop app can call it directly. `vastdb` (2.1.0) installs on macOS.

## How HandGuide uses it, in priority order

1. **Slow loop on the event's Cosmos3-Reason (about 30 min, must do).** Copy `GPU_BEARER_TOKEN` from the VM's team config into `~/RTHACK/.env.event` (gitignored; never commit it).
   - First measure accuracy and latency:
     ```bash
     python bench/cosmos_eval.py --url http://166.19.38.112:8001/v1 --model nvidia/cosmos3-reason \
         --tag event_cosmos3 --json-mode schema
     ```
     Unknowns: whether Cosmos 3 keeps 0–1000 boxes, whether it accepts `json_schema`, and whether the affordance hint is still needed. If schema mode is rejected, add `--json-mode prompt`.
   - Then run live with `--cosmos-url http://166.19.38.112:8001/v1 --cosmos-model nvidia/cosmos3-reason --cosmos-key $GPU_BEARER_TOKEN`.
2. **"Where did I put it?" scene memory on VastDB (about 2 h; this is the VAST story).**
   - Every slow-loop call already returns a description of the newest frame.
   - Embed it with Cosmos Embed1 (256-d). Write a row to **our own table** in the team's VastDB bucket: time, description, target, box, an S3 key for a thumbnail, and the 256-d vector.
   - A question such as "where's my mug?" runs `vector_search`, which returns the last place the mug was seen. That becomes the target, and the fast loop guides the hand to it.
   - This is the demo story: video memory search on VAST, followed by real-time action.
3. **Voice command with Canary-1B (about 1 h, optional).** A blind user says "find my cup" instead of typing. This needs microphone permission on the Mac.
4. **A W&B LLM for intent (small).** It maps "I want coffee" to `cup`, or "go outside" to `door handle`.

## Rules to respect (from the starter repo)

- **Stay inside the team's credentials, buckets and namespace.**
- **Don't modify or redeploy the shared DataEngine pipeline, and don't write into `vss-collection`.** Make our own schema/table, for example `handguide`.
- **Don't put files in the team chunks bucket.** The pipeline ingests from it, and the organizers say not to upload new chunks. Ask them where scratch objects may go, or keep thumbnails out of S3.
- **Don't ingest internet video.**
