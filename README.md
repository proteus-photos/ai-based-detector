[![Zenodo DOI](https://zenodo.org/badge/843042013.svg)](https://doi.org/10.5281/zenodo.15525425) [![arXiv](https://img.shields.io/badge/arXiv-10.48550/arXiv.2503.11195-b31b1b.svg)](https://doi.org/10.48550/arXiv.2503.11195))

This is the deep learning based detector tool that determines if an image is AI generated or not. This work outperforms previous open source work including LAANet, Corvi 2023, and Cozzolino 2023. It is used as a backup in the [Proteus](https://proteus.photos) provenance system.

This is the project data structure that the models expect.
```
$PROJECT_ROOT/data
    - real/
        - raise
    - ai-gen/
        - dalle2
        - dalle3
        - midjourney-v5
        - firefly
```
