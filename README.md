![license](https://img.shields.io/badge/license-Apache%202-green?style=flat)
[![data](https://img.shields.io/badge/data-Zenodo-blue?style=flat)](https://zenodo.org/records/20544289)
[![paper](https://img.shields.io/badge/paper-ACL_Anthology-red?style=flat)](https://aclanthology.org/2026.emnlp-main.2667/)

## Project Structure

- **`ouvia/`** – Flask web application that powers the study portal. Handles user authentication, role-based dashboards, conversation entry management, assignment workflows, and the recording/annotation pipeline.
- **`study/`** – Analysis and result-generation code for the paper. Contains data preparation scripts, statistical analyses, quality checks, and automatic translation quality evaluation.

> [!NOTE]
> We will soon release a basic documentation in the GitHub wiki as a tutorial walkthrough for running a new study.

> [!WARNING]
> This is a research project. All results reported in our paper and new studies can be reproduced with the code released here, but this is **not a production-ready platform**. Support is provided on a best-effort basis — if anything is unclear, please [open an issue](https://github.com/g8a9/ouvia/issues) and we'll do our best to help.

## Data Release

We release all the study artifacts, including raw audio recordings, machine translation outputs, automatic quality metrics, and human survey responses collected during the study on [Zenodo](https://zenodo.org/records/20544289).

## License

The code in this repository is licensed under Apache 2.0. All data artifacts are released under CC BY 4.0.

## Citation

```bibtex
@inproceedings{attanasio-etal-2026-ouvia,
    title = "Ouvia: A User-centered Framework for Measuring Usability of Speech Translation in Real-World Communication Scenarios",
    author = "Attanasio, Giuseppe  and
      Savoldi, Beatrice  and
      Chechelnitsky, Daniel  and
      Negri, Matteo  and
      Carpuat, Marine  and
      Sap, Maarten  and
      Martins, Andre",
    editor = "Lee, Hung-Yi  and
      Dev, Sunipa  and
      Li, Jessy  and
      Elliott, Desmond",
    booktitle = "Proceedings of the 2026 Conference on Empirical Methods in Natural Language Processing",
    month = oct,
    year = "2026",
    address = "Budapest, Hungary",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2026.emnlp-main.2667/",
    pages = "60698--60716",
    ISBN = "979-8-89176-459-0",
    abstract = "Speech translation (ST) is increasingly adopted in user applications, yet its evaluation largely focuses on decontextualized testbeds and holistic quality, rather than end users' communication needs. We introduce Ouvia, an evaluation framework for measuring user-perceived usability of speech translation outputs in real-world settings. Ouvia focuses on one-to-one communication: an English speaker needs to convey a request to a Portuguese speaker, and the message is automatically translated. Through a custom web app and multi-phase study design, we collect more than 1,750 such interactions in healthcare and everyday situations, mediated by four ST systems, involving speakers from three English dialects and two genders. We find that modern ST serves people only to a limited extent{---}only around half of interactions are rated as usable{---}with significant gaps in reported usability across demographic groups. Moreover, among quality metrics, we find that QA-based evaluation is a substantially stronger predictor of real-world usability than standard approaches. Together, these findings stress the importance of situated, user-centered evaluation frameworks that go beyond holistic quality scores and attend to who the technology serves{---}and how well."
}
```
