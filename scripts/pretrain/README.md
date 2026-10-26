# Workshop pretraining

PyTorch twin of the Workshop GPT (`src/lab/engine/gpt.ts`), used to pretrain the Large, Huge and XL sizes on raw UTF-8 bytes and ship them as `public/models/<size>.bin`.

| file | what it does |
|---|---|
| `model.py` | exact replica of the TS engine: same op order, tanh-GELU, LayerNorm eps 1e-5, init, `[in, out]` weight layout, parameter order |
| `export.py` | writes a model or checkpoint as a `.bin` (fp16, or fp32 for tests) |
| `parity.py` + `parity_engine.ts` | the gate for pretraining: random weights in PyTorch, exported and run through the TS CPU engine in Node, logits and loss compared |
| `prepare_data.py` | builds the dataset (English + Swedish Project Gutenberg) |
| `train.py` | pretraining: bf16 autocast, `torch.compile`, AdamW, warmup + cosine, validation, checkpoints, export |

Everything big lives outside OneDrive: the venv at `C:\Users\elias\llm-pretrain\.venv`, the data at `...\llm-pretrain\data`, the runs at `...\llm-pretrain\runs`.

## Setup

```powershell
C:\Users\elias\AppData\Local\Programs\Python\Python312\python.exe -m venv C:\Users\elias\llm-pretrain\.venv
C:\Users\elias\llm-pretrain\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Parity test (run it before any pretraining)

```powershell
$env:Path = "C:\Program Files\nodejs;" + $env:Path   # Node 22.18+ (runs the .ts directly)
C:\Users\elias\llm-pretrain\.venv\Scripts\python.exe parity.py
```

The targets are a max |logit diff| ≤ 1e-4 and a relative loss difference ≤ 1e-5 in fp32 export mode. For the GPU, load a `.bin` in the Workshop (or call `__gpuSelfTest('/models/<file>.bin')` in the dev console) and the self-test adds a case comparing the WebGPU engine with the CPU engine on those weights.

## Pretraining (B2)

```powershell
cd scripts\pretrain
$py = "C:\Users\elias\llm-pretrain\.venv\Scripts\python.exe"
& $py prepare_data.py                                   # about 420 MB into ...\data\pg-en-sv
& $py train.py --size large --export ..\..\public\models\large.bin
& $py train.py --size huge  --export ..\..\public\models\huge.bin
& $py train.py --size xl    --export ..\..\public\models\xl.bin
```

Measured B2 runs on the RTX 4080 SUPER, 2026-10-26, with `torch.compile` and bf16, on a GPU shared with other work:

| size | params | tokens | avg rate | wall time | final train / val loss (per byte) | file |
|---|---|---|---|---|---|---|
| Large | 1.90 M | 0.50 B | ~2.3 M tok/s | 4 min | 1.216 / 1.295 | 3.8 MB |
| Huge | 10.9 M | 1.00 B | ~0.6 to 0.74 M tok/s | ~33 min | 1.072 / 1.135 | 21.9 MB |
| XL | 25.6 M | 1.50 B | ~0.33 M tok/s | 81 min | 1.005 / 1.081 | 51.2 MB |

The Huge run hit a CUDA out-of-memory error at step 17,950, while another process held about 7 GB of VRAM. It was resumed from the step-16,000 checkpoint with `--batch 64 --accum 2`, which keeps the same tokens per step. Checkpoints are written atomically every 2,000 steps or 5 minutes, whichever comes first.

`--resume` continues from `runs\<size>\ckpt.pt`. `export.py --ckpt ... --out ...` re-exports any checkpoint.

## Weight file format (`.bin`, version 1)

All integers and floats are little-endian.

| bytes | content |
|---|---|
| 0 to 7 | magic `LLMXWGT\0` |
| 8 to 11 | `u32` format version (1) |
| 12 to 15 | `u32` header length *H*, padded with spaces so the data starts 16-byte aligned |
| 16 to 16+H | UTF-8 JSON header |
| 16+H to end | every parameter as `f16` (or `f32`), concatenated in `layout` order, each row-major |

Header fields:

- `format`: `"llm-explainer-gpt"`
- `size`: the Workshop size id (`large` / `huge` / `xl`)
- `config`: `{vocabSize, ctx, nLayer, nHead, dModel}`
- `vocab`: `{type: "bytes", size: 256}`, where token *i* is the byte value *i*
- `dtype`: `"f16"` or `"f32"`
- `layout`: `[{name, shape}]` in engine order: `tok [V,d]`, `pos [ctx,d]`, then for each layer `l` the tensors `l.ln1g`, `l.ln1b`, `l.wqkv [d,3d]` (columns q, then k, then v), `l.bqkv`, `l.wo [d,d]`, `l.bo`, `l.ln2g`, `l.ln2b`, `l.wfc [d,4d]`, `l.bfc`, `l.wproj [4d,d]`, `l.bproj`, and finally `lnfg`, `lnfb`, `wout [d,V]`, `bout`
- `numParams`
- `training`: `{dataset, datasetSource, tokensSeen, steps, finalValLoss, finalTrainLoss, peakLr, date, trainer}`

The loader is `src/lab/engine/weights.ts`. It checks the magic, the version, the layout against the engine, the exact file length and that every weight is finite, then converts to Float32. Anything without the magic counts as a missing file. That covers a 404, and also Vercel's SPA rewrite or Vite's dev fallback answering a missing `.bin` with `index.html`, which the page then reports as "Pretrained weights coming soon".

## Dataset and licences

- **Source:** [`manu/project_gutenberg`](https://huggingface.co/datasets/manu/project_gutenberg) on the Hugging Face Hub, streamed with `datasets`; nothing is fetched from gutenberg.org. Each row is one book with its original Project Gutenberg header and footer.
- **Built dataset (`pg-en-sv`, 2026-10-26):** the build used the Swedish split in full and shuffled English shards until it had 320 MB of English, after which collection stopped.
  - English: 810 books, 321 MB.
  - Swedish: 211 books, 60 MB. The split has 388 rows, but many are the same book in two encodings, so only 211 are unique.
  - Split: 378.1 MB train and 2.9 MB validation, with 4 English and 2 Swedish books held out.
  - Filtering: of 1,772 rows read, 654 were dropped as duplicates, 95 as unusable and 2 as Workshop texts.
- **Licence of the texts:** they are public-domain works in the United States, as distributed by Project Gutenberg. PG's own licence covers the "Project Gutenberg" trademark and the boilerplate text. `prepare_data.py` strips the header, footer, licence and credit lines, so what remains is the bare public-domain text, which can be used freely.
- **Licence of the HF dataset:** its card declares no licence, and the License section is empty. It only repackages the PG files, so we rely on the public-domain status of the texts themselves. A fully licensed alternative for English only is [`sedthh/gutenberg_english`](https://huggingface.co/datasets/sedthh/gutenberg_english) (MIT, PG metadata removed), but its text has doubled line breaks inside paragraphs, so it isn't used. [`deepmind/pg19`](https://huggingface.co/datasets/deepmind/pg19) (Apache-2.0) is script-based and English only.
- **Excluded:** the Workshop's own texts in any edition: Macbeth, Röda rummet and Grimm's fairy tales (by PG id and by title). This keeps fine-tuning on them a fair test of transfer.
- **Cleaning:** the following are dropped:
  - the PG header, footer and licence;
  - "Produced by", Gutenberg and URL notes at the start of each book;
  - `[Illustration]` lines;
  - books under 20 kB;
  - books that are mostly non-letters;
  - books with mis-decoded text;
  - exact duplicates, and near duplicates with the same normalised opening 4 kB (PG often has several editions of one book).

  Text is normalised to NFC with `\n` line endings.
- **Split:** whole books are held out, about 0.5% of each language's bytes, chosen by a hash of the book id. The output is `train.bin` and `val.bin` (raw bytes) plus `meta.json`, which lists every book and its split.
