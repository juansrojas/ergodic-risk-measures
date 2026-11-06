# Ergodic Risk Measures: Towards a Risk-Aware Foundation for Continual Reinforcement Learning

This repository contains the official implementation for

**Ergodic Risk Measures: Towards a Risk-Aware Foundation for Continual Reinforcement Learning**

*By: Juan Sebastian Rojas and Chi-Guhn Lee*

Conference on Neural Information Processing Systems (NeurIPS), 2026

**Abstract**: Continual reinforcement learning (continual RL) seeks to formalize the notions of lifelong learning and 
endless adaptation in RL. In particular, the aim of continual RL is to develop RL agents that can maintain a careful 
balance between retaining useful information and adapting to new situations. To date, continual RL has been explored 
almost exclusively through the lens of risk-neutral decision-making, in which the agent aims to optimize the expected 
long-run performance. In this work, we present the first formal theoretical treatment of continual RL through the lens 
of risk-aware decision-making, in which the behaviour of the agent is directed towards optimizing a measure of long-run 
performance beyond the mean. In particular, we show that the classical theory of risk measures, widely used as a 
theoretical foundation in non-continual risk-aware RL, is, in its current form, incompatible with continual learning. 
Then, building on this insight, we extend risk measure theory into the continual setting by introducing a new class of 
ergodic risk measures, and showing that it is compatible with continual learning. Finally, we provide a case study of 
continual risk-aware learning, along with empirical results, which show the intuitive appeal of ergodic risk measures 
in continual settings.

<p align="center">
  &#151; <a href="https://neurips.cc/virtual/2026/loc/sydney/poster/152958"><b>View Paper</b></a> &#151;
</p>

## Requirements

To install requirements:

```
pip install -r requirements.txt
```

## Results

To train the agents and reproduce the results in the paper, run the cells in the `Experiments.ipynb` notebook. An html
file with cell outputs that match our paper's results is provided for reference (`Experiments.html`). 

## Citation 

```
@inproceedings{rojas2026ergodic,
    title={Ergodic Risk Measures: Towards a Risk-Aware Foundation for Continual Reinforcement Learning},
    author={Rojas, Juan Sebastian and Lee, Chi-Guhn},
    booktitle={Advances in Neural Information Processing Systems 39 (NeurIPS 2026)},
    year={2026}
}
```

Please cite our paper if you use this code in your own work.