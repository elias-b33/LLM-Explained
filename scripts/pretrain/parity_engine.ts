/**
 * TS side of the parity test (run by parity.py; needs Node 22.18+ / 24 for
 * built-in TypeScript type stripping):
 *
 *   node scripts/pretrain/parity_engine.ts <weights.bin> <case.json> <out-logits.f32>
 *
 * Loads the weights with the Workshop's own loader, runs the Workshop's CPU
 * engine (Float32, exactly as in the training worker) on the token ids in
 * case.json, writes the logits as raw little-endian float32 and prints the loss
 * (plus the KV-cached decoder's last-row logits check) as JSON on stdout.
 */
import { readFileSync, writeFileSync } from 'node:fs'
import { Decoder } from '../../src/lab/engine/gpt.ts'
import { modelFromWeights, parseWeights } from '../../src/lab/engine/weights.ts'

const [binPath, casePath, outPath] = process.argv.slice(2)
const file = readFileSync(binPath)
const buf = file.buffer.slice(file.byteOffset, file.byteOffset + file.byteLength)
const t0 = performance.now()
const w = parseWeights(buf)
const loadMs = performance.now() - t0
const c = JSON.parse(readFileSync(casePath, 'utf8')) as { B: number; T: number; idx: number[]; targets: number[] }
const model = modelFromWeights(w)
const { B, T } = c
const V = w.header.config.vocabSize

// logits (forward without targets keeps raw logits), then the loss (forward with targets)
const t1 = performance.now()
model.forward(c.idx, null, B, T)
const logits = new Float32Array(B * T * V)
for (let n = 0; n < B * T; n++) logits.set(model.logitsRow(B, T, n), n * V)
const loss = model.forward(c.idx, c.targets, B, T)
const fwdMs = performance.now() - t1
writeFileSync(outPath, Buffer.from(logits.buffer))

// the generation path: KV-cached decoder fed row 0 token by token must give the same last-row logits
const dec = new Decoder(model)
dec.prime(c.idx.slice(0, T))
let decMax = 0
for (let i = 0; i < V; i++) decMax = Math.max(decMax, Math.abs(dec.logits[i] - logits[(T - 1) * V + i]))

console.log(JSON.stringify({ loss, numParams: model.numParams, dtype: w.header.dtype, decoderVsForwardMaxAbs: decMax, loadMs, fwdMs }))
