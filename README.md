# Language-TPP

Source code for paper "Language-TPP: Integrating Temporal Point Processes with Language Models for Event Analysis"

## Overview

Please refer to the following script for a demo of the proposed model:
- `Language_TPP_0___5B/modeling_langtpp.py`: revised version of the original Qwen2 model with the computation of log-likelihood values of TPPs. Note that pretrained model weights are not included in this repository and we plan to release them soon.
- `demo.py`: demo script for the revised Qwen2 model, along with the proposed byte tokens and inference process.

## Quick Start

Use [uv](https://docs.astral.sh/uv/) to set up the environment:
```sh
# On macOS and Linux.
curl -LsSf https://astral.sh/uv/install.sh | sh

uv sync
```


## TODO

- [ ] Release pretrained model weights
- [ ] Release training scripts and documentation
- [ ] Release evaluation scripts and benchmarks


## Acknowledgments

This codebase builds upon the architecture and implementation of [Qwen2](https://github.com/huggingface/transformers/tree/v4.46.3/src/transformers/models/qwen2).


## Citation

If you find this work useful in your research, please consider citing our paper:

```bibtex
@article{kong2024languagetpp,
  title={Language-TPP: Integrating Temporal Point Processes with Language Models for Event Analysis},
  author={Kong, Quyu and Zhang, Yixuan and Liu, Yang and Tong, Panrong and Liu, Enqi and Zhou, Feng},
  journal={arXiv preprint},
  year={2024}
}