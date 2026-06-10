![license](https://img.shields.io/badge/license-Apache%202-green?style=flat)
[![data](https://img.shields.io/badge/data-Zenodo-blue?style=flat)](https://zenodo.org/records/20544289)
[![arXiv](https://img.shields.io/badge/arXiv-2606.06177-red?style=flat)](https://arxiv.org/abs/2606.06177)

## Project Structure

- **`ouvia/`** – Flask web application that powers the study portal. Handles user authentication, role-based dashboards, conversation entry management, assignment workflows, and the recording/annotation pipeline.
- **`study/`** – Analysis and result-generation code for the paper. Contains data preparation scripts, statistical analyses, quality checks, and automatic translation quality evaluation.

> [!INFO]
> We will soon release a basic documentation in the GitHub wiki as a tutorial walkthrough for running a new study.

> [!IMPORTANT]
> This is a research project. All results reported in our paper and new studies can be reproduced with the code released here, but this is **not a production-ready platform**. Support is provided on a best-effort basis — if anything is unclear, please [open an issue](https://github.com/g8a9/ouvia/issues) and we'll do our best to help.

## Data Release

We release all the study artifacts, including raw audio recordings, machine translation outputs, automatic quality metrics, and human survey responses collected during the study on [Zenodo](https://zenodo.org/records/20544289).

## License

The code in this repository is licensed under Apache 2.0. All data artifacts are released under CC BY 4.0.

## Citation

```bibtex
@article{attanasio2026ouvia,
  title={Ouvia: A User-centered Framework for Measuring Usability of Speech Translation in Real-World Communication Scenarios},
  author={Attanasio, Giuseppe and Savoldi, Beatrice and Chechelnitsky, Daniel and Negri, Matteo and Carpuat, Marine and Sap, Maarten and Martins, Andr{\'e} FT},
  journal={arXiv preprint arXiv:2606.06177},
  year={2026}
}
```