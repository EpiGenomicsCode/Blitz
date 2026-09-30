## Blitz

Blitz distills the Boltz-2 structure model into deterministic 8-step and
16-step students. This branch contains the code for training and running both
models.

On [Boltz's roughly 2,300-target benchmark](https://doi.org/10.1101/2025.06.14.659707):

| system | model | NFE | complex lDDT ↑ | RF-valid ↑ | DockQ ↑ | ligand RMSD ↓ | any violation ↓ |
|---|---|---:|---:|---:|---:|---:|---:|
| AlphaFold 3 | teacher | 200 | 0.8615 | 41.5% | 0.4198 | 6.63 Å | 9.8% |
| AlphaFold 3 | Blitz K8 | 8 | **0.8635** | 40.5% | **0.4207** | **6.44 Å** | 11.3% |
| AlphaFold 3 | Blitz K16 | 16 | **0.8634** | **42.3%** | 0.4167 | **6.51 Å** | **6.8%** |
| Boltz-2 | teacher | 600 | 0.8499 | 98.5% | 0.3929 | 9.00 Å | 30.3% |
| Boltz-2 | Blitz K8 | 8 | 0.8492 | 93.7% | 0.3897 | **8.75 Å** | 41.9% |
| Boltz-2 | Blitz K16 | 16 | **0.8533** | 94.1% | **0.3932** | **8.78 Å** | **26.7%** |

Every row uses the same five-candidate reranker. Bold values are student point
estimates that improve on the corresponding teacher. The AF3 results show the
same overall pattern: teacher-level structural accuracy at a small fraction of
the sampling cost, with K16 also improving the violation rate. NFE counts
denoiser evaluations; the Boltz-2 teacher uses three recycle updates.

AlphaFold 3 checkpoints are not included because Google DeepMind's terms do
not permit public redistribution of the model parameters.

Download the standalone inference checkpoints from
[`vinaymatt/Blitz-Boltz2`](https://huggingface.co/vinaymatt/Blitz-Boltz2).

K16 is the best default. K8 trades some validity for fewer denoising steps.

See the [quickstart](docs/blitz/QUICKSTART.md) to run the models. The
[training](docs/blitz/TRAINING.md), [data](docs/blitz/DATA.md), and
[results](docs/blitz/RESULTS.md) pages describe how they were produced and
evaluated. [Model lineage](docs/blitz/PROVENANCE.md) explains how this branch
relates to upstream Boltz. The original Boltz README follows.

---

<div align="center">
  <div>&nbsp;</div>
  <img src="docs/boltz2_title.png" width="300"/>
  <img src="https://model-gateway.boltz.bio/a.png?x-pxid=bce1627f-f326-4bff-8a97-45c6c3bc929d" />

[Boltz-1](https://doi.org/10.1101/2024.11.19.624167) | [Boltz-2](https://doi.org/10.1101/2025.06.14.659707) |
[Slack](https://boltz.bio/join-slack) <br> <br>
</div>



![](docs/boltz1_pred_figure.png)


## Introduction

Boltz is a family of models for biomolecular interaction prediction. Boltz-1 was the first fully open source model to approach AlphaFold3 accuracy. Our latest work Boltz-2 is a new biomolecular foundation model that goes beyond AlphaFold3 and Boltz-1 by jointly modeling complex structures and binding affinities, a critical component towards accurate molecular design. Boltz-2 is the first deep learning model to approach the accuracy of physics-based free-energy perturbation (FEP) methods, while running 1000x faster — making accurate in silico screening practical for early-stage drug discovery.

All the code and weights are provided under MIT license, making them freely available for both academic and commercial uses. For more information about the model, see the [Boltz-1](https://doi.org/10.1101/2024.11.19.624167) and [Boltz-2](https://doi.org/10.1101/2025.06.14.659707) technical reports. To discuss updates, tools and applications join our [Slack channel](https://boltz.bio/join-slack).

## Installation

> Note: we recommend installing boltz in a fresh python environment

Install boltz with PyPI (recommended):

```
pip install boltz[cuda] -U
```

or directly from GitHub for daily updates:

```
git clone https://github.com/jwohlwend/boltz.git
cd boltz; pip install -e .[cuda]
```

If you are installing on CPU-only or non-CUDA GPus hardware, remove `[cuda]` from the above commands. Note that the CPU version is significantly slower than the GPU version.

## Inference

You can run inference using Boltz with:

```
boltz predict input_path --use_msa_server
```

`input_path` should point to a YAML file, or a directory of YAML files for batched processing, describing the biomolecules you want to model and the properties you want to predict (e.g. affinity). To see all available options: `boltz predict --help` and for more information on these input formats, see our [prediction instructions](docs/prediction.md). By default, the `boltz` command will run the latest version of the model.


### Binding Affinity Prediction
There are two main predictions in the affinity output: `affinity_pred_value` and `affinity_probability_binary`. They are trained on largely different datasets, with different supervisions, and should be used in different contexts. The `affinity_probability_binary` field should be used to detect binders from decoys, for example in a hit-discovery stage. Its value ranges from 0 to 1 and represents the predicted probability that the ligand is a binder. The `affinity_pred_value` aims to measure the specific affinity of different binders and how this changes with small modifications of the molecule. This should be used in ligand optimization stages such as hit-to-lead and lead-optimization. It reports a binding affinity value as `log10(IC50)`, derived from an `IC50` measured in `μM`. More details on how to run affinity predictions and parse the output can be found in our [prediction instructions](docs/prediction.md).

## Authentication to MSA Server

When using the `--use_msa_server` option with a server that requires authentication, you can provide credentials in one of two ways. More information is available in our [prediction instructions](docs/prediction.md).
 
## Evaluation

⚠️ **Coming soon: updated evaluation code for Boltz-2!**

To encourage reproducibility and facilitate comparison with other models, on top of the existing Boltz-1 evaluation pipeline, we will soon provide the evaluation scripts and structural predictions for Boltz-2, Boltz-1, Chai-1 and AlphaFold3 on our test benchmark dataset, and our affinity predictions on the FEP+ benchmark, CASP16 and our MF-PCBA test set.

![Affinity test sets evaluations](docs/pearson_plot.png)
![Test set evaluations](docs/plot_test_boltz2.png)


## Training

⚠️ **Coming soon: updated training code for Boltz-2!**

If you're interested in retraining the model, currently for Boltz-1 but soon for Boltz-2, see our [training instructions](docs/training.md).


## Contributing

We welcome external contributions and are eager to engage with the community. Connect with us on our [Slack channel](https://boltz.bio/join-slack) to discuss advancements, share insights, and foster collaboration around Boltz-2.

On recent NVIDIA GPUs, Boltz leverages the acceleration provided by [NVIDIA  cuEquivariance](https://developer.nvidia.com/cuequivariance) kernels. Boltz also runs on Tenstorrent hardware thanks to a [fork](https://github.com/moritztng/tt-boltz) by Moritz Thüning.

## License

Our model and code are released under MIT License, and can be freely used for both academic and commercial purposes.


## Cite

If you use this code or the models in your research, please cite the following papers:

```bibtex
@article{passaro2025boltz2,
  author = {Passaro, Saro and Corso, Gabriele and Wohlwend, Jeremy and Reveiz, Mateo and Thaler, Stephan and Somnath, Vignesh Ram and Getz, Noah and Portnoi, Tally and Roy, Julien and Stark, Hannes and Kwabi-Addo, David and Beaini, Dominique and Jaakkola, Tommi and Barzilay, Regina},
  title = {Boltz-2: Towards Accurate and Efficient Binding Affinity Prediction},
  year = {2025},
  doi = {10.1101/2025.06.14.659707},
  journal = {bioRxiv}
}

@article{wohlwend2024boltz1,
  author = {Wohlwend, Jeremy and Corso, Gabriele and Passaro, Saro and Getz, Noah and Reveiz, Mateo and Leidal, Ken and Swiderski, Wojtek and Atkinson, Liam and Portnoi, Tally and Chinn, Itamar and Silterra, Jacob and Jaakkola, Tommi and Barzilay, Regina},
  title = {Boltz-1: Democratizing Biomolecular Interaction Modeling},
  year = {2024},
  doi = {10.1101/2024.11.19.624167},
  journal = {bioRxiv}
}
```

In addition if you use the automatic MSA generation, please cite:

```bibtex
@article{mirdita2022colabfold,
  title={ColabFold: making protein folding accessible to all},
  author={Mirdita, Milot and Sch{\"u}tze, Konstantin and Moriwaki, Yoshitaka and Heo, Lim and Ovchinnikov, Sergey and Steinegger, Martin},
  journal={Nature methods},
  year={2022},
}
```
