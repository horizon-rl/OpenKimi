# OpenKimi

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg?style=flat-square)](LICENSE)
[![arXiv](https://img.shields.io/badge/arXiv-2602.05933-b31b1b.svg?style=flat-square)](https://arxiv.org/abs/2602.05933)


OpenKimi is a research project that implements the reinforcement learning (RL) algorithms and efficient rollout system used in [Kimi-K2](https://github.com/MoonshotAI/Kimi-K2) and [Kimi-K1.5](https://github.com/MoonshotAI/Kimi-k1.5) by @MoonshotAI. Fascinated by the strong performance of Kimi-series models, this project provides theoretical understanding of Policy Mirror Descent (PMD) algorithms that differ from other policy gradient methods like GRPO, along with practical training recipes that achieve superior performance and time efficiency for various downstream tasks. OpenKimi enables both algorithmic exploration for RL fine-tuning and system efficiency research for accelerating rollout in asynchronous RL.  

## Quick Start

### Preparation

```bash
git clone --recurse-submodules https://github.com/horizon-rl/OpenKimi.git
cd OpenKimi && cd verl && pip install -e .
```
### Training Examples

#### Math Reasoning

We provide training example scripts for both FSDP (for smaller models) and Megatron (for larger models and MoE) backends. More details are available in [openkimi/pmd/README.md](openkimi/pmd/README.md).

```bash
bash examples/math/run_pmd_dapo17k_qwen25-7b.sh
```

#### Upcoming Features

We are actively developing additional features and training recipes:

**System Enhancements**
- [ ] **Hybrid Partial Rollout**
- [ ] **Muon Optimizer**

**Recipes**
- [ ] **Advanced Mismatch Correction**
- [ ] **Tool-Integrated Reasoning**
- [ ] **Kimi-K2 Agentic RL Training**


## Contributing

**Contributions are welcome!** If you have suggestions or feedback on new features and recipes, feel free to submit an issue or pull request. 

```bash
pre-commit install

pre-commit run --all-files --show-diff-on-failure --color=always
```

## 📖 Citation

```
@misc{xu2026pmdkimi,
      title={Approximation of Log-Partition Function in Policy Mirror Descent Induces Implicit Regularization for LLM Post-Training}, 
      author={Zhenghao Xu, Qin Lu, Changlong Yu and Tuo Zhao},
      year={2026},
      eprint={2602.05933},
      archivePrefix={arXiv},
      primaryClass={cs.ML},
}
```