# RLIonS: Reinforcement Learning for Ion Shuttling

This is the official implementation of our proposed algorithm from the paper "Reinforcement learning for ion shuttling on trapped-ion quantum computers".

## License
This project is dual-licensed:
* All data contained within the `/Data` directory is licensed under **Creative Commons Attribution 4.0 International (CC-BY-4.0)**. See [Data/LICENSE.md](Data/LICENSE.md) for details.
* All other files, including but not limited to all source code, is licensed under the **GNU Affero General Public License v3.0**, see [License.txt](License.txt), Copyright 2026 Maximilian Schier, Lea Richtmann.

## Installation

The installation instructions use either an already configured conda or Miniforge (mamba) installtion.
Either choice works, the instructions below assume `mamba` is used. Please consult the installation
instructions of either `conda` or `mamba` if you have neither already installed.

```shell
mamba create -n ionshuttle python=3.10.14 pycairo=1.29.0
mamba activate ionshuttle
# EITHER: On Linux to install with CUDA support
pip install -r requirements.txt
# OR: On Windows or other systems without CUDA support
pip install -r requirements.win.txt
```

## Training
To train an RLIonS compiler, run `rlions_train.py`. The important commands to replicate experiments from the paper are:
```bash
# Agents for main expriments for three architectures
PYTHONPATH=. python ppo_example.py --chip qvls_x_50
PYTHONPATH=. python ppo_example.py --chip qvls_q_50
PYTHONPATH=. python ppo_example.py --chip qvls_q_50_s3
# Agents for ablation experiments
PYTHONPATH=. python ppo_example.py --tag qv50x_linear --chip qvls_x_50 --obs_number_encoder linear
PYTHONPATH=. python ppo_example.py --tag qv50x_noshaped --chip qvls_x_50 --factor_completion 0
PYTHONPATH=. python ppo_example.py --tag qv50x_same_gamma --chip qvls_x_50 --shaped_gamma 0.9995
PYTHONPATH=. python ppo_example.py --tag qv50x_basic --chip qvls_x_50 --obs_encoder basic
```

## Compiling QV and MQT circuits
To compile the QV and MQT circuits, run `rlions_compile.py` with a previously trained agent. To compile the QV(6)
problems over an extended time frame, run `rlions_compile_long.py`.

## Results and plotting
The results of the paper are contained in the `Data` folder. To recreate the evaluation from the experiments, the
following scipts can be used:

* `eval_plot_qv.py`: Plot main and ablation figures for the QV experiments.
* `eval_table_mqt.py`: Make tables (and plots) for the MQT experiments.
* `eval_table_qv_sat_opt.py`: Make tables for the optimality gap experiments.

## Citing

If you find our work useful, consider citing our paper:
```
@article{b7ck-8wh4,
  title = {Reinforcement learning for ion shuttling on trapped-ion quantum computers},
  author = {Schier, Maximilian and Richtmann, Lea and Staufenbiel, Christian and Schmale, Tobias and Borcherding, Daniel and Heurs, Michèle and Rosenhahn, Bodo},
  journal = {Phys. Rev. Res.},
  pages = {},
  year = {2026},
  month = {Aug},
  publisher = {American Physical Society},
  doi = {10.1103/b7ck-8wh4},
  url = {https://link.aps.org/doi/10.1103/b7ck-8wh4}
}
```