# MicDrop dataset evaluation report

Generated: 2026-09-25T13:02:10.280887+00:00  (rng seed 20260925)

## Totals

| split | examples |
|---|---|
| train | 1095 |
| validation | 152 |
| test | 153 |
| **total** | **1400** |

## Examples per category

| category | train | validation | test |
|---|---|---|---|
| clean_simple | 55 | 5 | 6 |
| commands | 6 | 2 | 0 |
| file_paths | 3 | 0 | 0 |
| filler_removal | 349 | 51 | 49 |
| long_requests | 0 | 1 | 1 |
| mixed_noise | 9 | 1 | 0 |
| multi_part | 11 | 2 | 0 |
| numbers_versions | 25 | 6 | 12 |
| programming | 51 | 6 | 3 |
| questions | 4 | 0 | 1 |
| real_recording | 1 | 0 | 1 |
| repetition | 194 | 30 | 27 |
| speech_correction | 220 | 19 | 31 |
| speech_style_normalization | 45 | 6 | 5 |
| technical_correction | 122 | 23 | 17 |

## Examples per source

| source | train | validation | test |
|---|---|---|---|
| codealpaca | 226 | 33 | 34 |
| dolly | 205 | 38 | 42 |
| dowis | 45 | 6 | 5 |
| magicoder | 149 | 19 | 22 |
| micdrop_real | 1 | 0 | 1 |
| micdrop_seed | 240 | 24 | 15 |
| oasst1 | 229 | 32 | 34 |

## Synthetic vs real-seed clean

- **train**: synthetic=1021, real_seed_clean=74
- **validation**: synthetic=147, real_seed_clean=5
- **test**: synthetic=146, real_seed_clean=7

## Duplicates removed (across all records before split)

- total: 2494
- dropped_exact_duplicate: 44
- kept: 2338
- dropped_near_duplicate_variant: 112
- final: 2338

## Length statistics (characters)

| split | input mean | input p95 | output mean | output p95 |
|---|---|---|---|---|
| train | 91.3 | 190 | 77.1 | 173 |
| validation | 90.2 | 182 | 77.5 | 167 |
| test | 97.1 | 195 | 82.7 | 174 |

## Longest inputs

- `oasst1`: Write a short story about how Adder a man no write a short story about how Adder a man with the power of teleportation films a YouTube prank video wherein he as...
- `oasst1`: I want you to no wait to act like an assistant for for for my own website You should be able uhm to answer questions regarding my skills and experience I will p...
- `oasst1`: Please discuss the point of view on how the democratization of no please discuss the point of view on how the democratization of AI can actually allow people to...

## Shortest inputs

- `dolly`: What is a bank
- `dolly`: What is fish oil
- `dolly`: Can we be lovers
