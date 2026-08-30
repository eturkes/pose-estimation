# scout-m2u4 — real-corpus timebase probe

Redacted aggregates only: no filename, path, or subject identifier. All 9 units resolved.

## U1-U5 — measured, per stratified sample

| # | measurement | n assets | statistic | value | script |
| - | ----------- | -------- | --------- | ----- | ------ |
| U1 | header `fps` vs `nominal_fs(rounded decode ts)` — relative error | 10 pilot assets | per-stratum median / max absolute relative error | iPad5/16.6=0.000107% / 0.000697%; iPad5/16.7=0.000276% / 0.001068%; Air-M2/18.1.1=0.000285% / 0.000405%; Air-M2/26.5=0.000560% / 0.000560% | inline pilot; final script follows |
| U2 | header `fps` vs `1/median(diff(rounded decode ts))` — relative error | 10 pilot assets | per-stratum median / max absolute relative error | iPad5/16.6=0.1493% / 0.1515%; iPad5/16.7=0.1539% / 0.4262%; Air-M2/18.1.1=0.2160% / 0.2181%; Air-M2/26.5=0.2165% / 0.2165% | inline pilot; final script follows |
| U3 | `trajectory_grid_status` residual under `nominal_fs` | 10 pilot assets | whole-clip median / max; maximum 1.0 s-window residual | iPad5/16.6=0.0302 / 0.0461; 0.0510. iPad5/16.7=0.0439 / 0.0457; 0.0592. Air-M2/18.1.1=0.0419 / 0.0496; 0.0497. Air-M2/26.5=0.0264 / 0.0264; 0.0488 | inline pilot; final script follows |
| U4 | `trajectory_grid_status` residual under `1/median(diff)` | 10 pilot assets; 610 one-second windows | per-stratum whole-clip median / max; window pass / fail | iPad5/16.6=0.4985 / 0.4985; 227 / 0. iPad5/16.7=0.4940 / 0.4955; 75 / 80. Air-M2/18.1.1=0.4925 / 0.4985; 133 / 0. Air-M2/26.5=0.4925 / 0.4985; 95 / 0 | `scripts/probe_timebase_grid.py --sample-size 10 --seed 2404` |
| U5 | share of assets whose decode clock falls back to `frame_idx/fps` vs uses POS_MSEC | 10 pilot assets; 13,043 frames | asset clock-source class; fallback-frame share | POS_MSEC-only=10/10; mixed=0/10; fallback-only=0/10; fallback frames=0/13,043 | `scripts/probe_timebase_grid.py --sample-size 10 --seed 2404` |

## U6 — grid feasibility verdict

| question | verdict | evidence |
| -------- | ------- | -------- |
| does a per-video single `fs` place a real clip's timestamps on a grid within `GRID_SLOT_TOLERANCE`? | pilot: yes with `nominal_fs`; no with median-diff over whole clips | `nominal_fs`: 10/10 clips + 610/610 one-second windows pass. Median-diff: 0/10 clips + 530/610 windows pass. Header: 10/10 + 610/610 pass. |
| does that answer differ by device configuration? | pilot: no for `nominal_fs`; median-diff window failure appears in iPad5/16.7 | Every stratum passes all `nominal_fs` whole clips and windows. Median-diff whole clips fail in every stratum; 80/155 iPad5/16.7 windows fail. |
| does clip length change it (residual vs duration)? | pilot: not for `nominal_fs`; strongly for median-diff | Spearman duration↔whole residual: `nominal_fs`=-0.091; header=-0.042; median-diff=0.912. |

## U7 — worst cases

Full 379-asset census; residual = per-file maximum across exporter-rounded 1 s windows under `nominal_fs`.

| rank | stratum | residual | duration_s | what breaks | remaining limit |
| ---- | ------- | -------- | ---------- | ----------- | --------------- |
| 1 | iPad5/16.7 | 0.0592165802499 | 40.7521 | nominal=80/80; legacy=0/80 windows | asset identity redacted; pose/export row loss not replayed |
| 2 | iPad5/16.6 | 0.0510246486046 | 24.5450 | nominal=48/48; legacy=48/48 windows | asset identity redacted; pose/export row loss not replayed |
| 3 | iPad5/16.7 | 0.0510233222275 | 21.0100 | nominal=41/41; legacy=41/41 windows | asset identity redacted; pose/export row loss not replayed |
| 4 | iPad5/16.7 | 0.0510231677635 | 21.0767 | nominal=41/41; legacy=41/41 windows | asset identity redacted; pose/export row loss not replayed |
| 5 | iPad5/16.6 | 0.0510207848784 | 24.3783 | nominal=47/47; legacy=47/47 windows | asset identity redacted; pose/export row loss not replayed |
| 6 | iPad5/16.7 | 0.0510166358595 | 24.3450 | nominal=47/47; legacy=47/47 windows | asset identity redacted; pose/export row loss not replayed |
| 7 | iPad5/16.6 | 0.0510151187905 | 27.7800 | nominal=54/54; legacy=54/54 windows | asset identity redacted; pose/export row loss not replayed |
| 8 | iPad5/16.6 | 0.0510092863060 | 20.6433 | nominal=40/40; legacy=40/40 windows | asset identity redacted; pose/export row loss not replayed |

## U8 — sampling frame

| item | value |
| ---- | ----- |
| population + strata | population=379 canonical; device configurations=125 iPad5/16.6 + 131 iPad5/16.7 + 57 Air-M2/18.1.1 + 66 Air-M2/26.5; codecs=256 h264 + 123 hevc; rotations=341×0° + 27×90° + 10×180° + 1×270° |
| sample size + selection rule | n=60; proportional device-config quotas + SHA-256(seed,purpose,asset_id) rank; mandatory coverage=all device/codec/rotation strata + nearest 29.963/29.987 Hz + maximum-fps asset. Sample strata: device=20/20/9/11 in population-table order; codec=40 h264 + 20 hevc; rotation=52×0° + 4×90° + 3×180° + 1×270°. |
| determinism (seed/order) | PASS; seed=2404; two absent-output runs; `cmp` rc=0; both SHA-256=`11f0076f779241464eff529cef21c76168fa8f1382f658360d5e2d96cd60d1c5`; generator digest embedded + stale-source refusal rc=2 verified |
| wall time | sample run 1=176.58 s / 266,972 KiB; sample rerun=288.53 s / 260,608 KiB under concurrent decode; full census=1,370.01 s / 278,756 KiB |

## U9 — register

| id | verdict | evidence | disposition |
| -- | ------- | -------- | ----------- |
| R01 | BLOCKER — P20 header clause false | sample=1/60 above `1e-4`, max=1.10035234739e-4; census=4/379, max=1.46938338724e-4; failures span both codecs + 3 device configs | MAIN must choose: A=retain P06 bound and encode P20 failure; B=amend the real-header clause while retaining synthetic P06. Outcome-based reselection is rejected. |
| R02 | PASS — P20 grid clause | sample nominal=3,200/3,200; census nominal=21,651/21,651; census maximum residual=0.0592165802499 < 0.25 | Accept the adopted estimator's real-corpus grid evidence. |
| R03 | PASS — estimator comparison | nominal header error ≤ legacy on 379/379 assets; legacy grid=21,571/21,651; one 119.97 fps asset accounts for all 80 legacy failures | Adoption remains evidence-backed despite R01's overbroad header claim. |
| R04 | LIMIT — census assurance | committed sample=60 + byte-identical rerun; full census=one redacted ephemeral run | No claim of full-census byte determinism; sample artifact owns P20's durable evidence. |
| R05 | LIMIT — scope + provenance | probe rounds every decoded non-empty frame; pose detection/export row loss is absent. `source_sha256` binds generator bytes only per frozen schema. | Claims cover decode timebase, not detector sparsity. Regeneration requires the canonical corpus + inventory + qualification inputs. |

SCOUT-M2U4-2-DONE-1
