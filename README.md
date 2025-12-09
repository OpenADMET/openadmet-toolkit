OpenADMET Toolkit
==============================
[//]: # (Badges)
[![Logo](https://img.shields.io/badge/OSMF-OpenADMET-%23002f4a)](https://openadmet.org/)
[![GitHub Actions Build Status](https://github.com/OpenADMET/openadmet_toolkit/workflows/CI/badge.svg)](https://github.com/OpenADMET/openadmet_toolkit/actions?query=workflow%3ACI)
[![codecov](https://codecov.io/gh/OpenADMET/openadmet_toolkit/branch/main/graph/badge.svg)](https://codecov.io/gh/OpenADMET/openadmet_toolkit/branch/main)
[![Documentation Status](https://readthedocs.org/projects/openadmet-toolkit/badge/?version=latest)](https://openadmet-toolkit.readthedocs.io/en/latest/?badge=latest)
[![Google Colab](https://img.shields.io/badge/Google%20Colab-F9AB00?logo=googlecolab&logoColor=fff)](https://try.openadmet.org)

`openadmet-toolkit` contains helpful toolkit functionality for the large scale ADMET modelling conducted as part of the [OpenADMET project](https://openadmet.org).
Much of the functionality is focused around data scraping, cleaning, screening and preparation for downstream machine learning workflows. Additional one-off workflows related to functionality used directly by the OpenADMET team are also contained in this repo. You can see a demonstration of some toolkit functionality in our [Example Tutorials](https://demos.openadmet.org) and [Google Colab](https://try.openadmet.org). More developed functionality for machine learning is contained in the `OpenADMET Models` [repo](https://github.com/OpenADMET/openadmet-models/tree/main).


>[!NOTE]
> This repo is under very active development and should be considered **not production ready**. Use at your own risk!


## License

This library is made available under the [MIT](https://opensource.org/licenses/MIT) open source license.


## Install

### Development version

The development version of `openadmet-toolkit` can be installed directly from the `main` branch of this repository.

First install the package dependencies using `mamba`:

```bash
mamba env create -f devtools/conda-envs/openadmet_toolkit.yaml
```

The `openadmet-toolkit` library can then be installed via:

```
python -m pip install -e --no-deps .
```


## Authors

The OpenADMET development team.


### Copyright

Copyright (c) 2025, OpenADMET Models Contributors


## Acknowledgements

OpenADMET is an [Open Molecular Software Foundation](https://omsf.io/) hosted project.
