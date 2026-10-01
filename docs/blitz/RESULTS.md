# Boltz benchmark results

We evaluate on the roughly 2,300-target benchmark introduced with
[Boltz-2](https://doi.org/10.1101/2025.06.14.659707); the paper describes the
benchmark construction and evaluation protocol. Each model generated five
candidates. Our reranker selects from those candidates without using the
reference structure. Every metric below uses that selection rule.

| system | model | NFE | complex lDDT ↑ | RF-valid ↑ | DockQ ↑ | ligand RMSD ↓ | any violation ↓ |
|---|---|---:|---:|---:|---:|---:|---:|
| AlphaFold 3 | teacher | 200 | 0.8615 | 41.5% | 0.4198 | 6.63 Å | 9.8% |
| AlphaFold 3 | Blitz K8 | 8 | **0.8635** | 40.5% | **0.4207** | **6.44 Å** | 11.3% |
| AlphaFold 3 | Blitz K16 | 16 | **0.8634** | **42.3%** | 0.4167 | **6.51 Å** | **6.8%** |
| Boltz-2 | teacher | 600 | 0.8499 | 98.5% | 0.3929 | 9.00 Å | 30.3% |
| Boltz-2 | Blitz K8 | 8 | 0.8492 | 93.7% | 0.3897 | **8.75 Å** | 41.9% |
| Boltz-2 | Blitz K16 | 16 | **0.8533** | 94.1% | **0.3932** | **8.78 Å** | **26.7%** |

Bold values are student point estimates that improve on the corresponding
teacher. Across both systems, K8 and K16 retain teacher-level complex accuracy
with far fewer denoiser evaluations. K16 is the strongest Boltz-2 default: it
improves complex lDDT, DockQ, ligand RMSD, and the overall violation rate over
the teacher. The AF3 comparison is similarly favorable, with both students
improving complex lDDT and K16 improving RF validity and geometric quality.

NFE counts denoiser evaluations. The Boltz-2 teacher uses three recycle
updates; all rows use the same five-candidate reranker.

Of these differences, the K16 violation-rate improvement is the clearest: 3.6
percentage points below the teacher, with a paired 95% interval of 1.9–5.3
points. The DockQ and ligand RMSD differences are smaller point improvements.

The checkpoint and sampling policy are a pair. Changing the schedule, endpoint
correction, candidate count, or selection rule produces a different result.
