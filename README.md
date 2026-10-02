# Attention-based Multiple Instance Learning (ABMIL) Core repository

## Status

This repository accompanies a manuscript currently under review. The codebase, documentation and dataset access instructions may be updated during the peer-review and publication process. Additional resources, including the full dataset and reproducibility scripts, will be finalized upon acceptance of the manuscript. 

## About
This repository integrates [**CLAM**](https://github.com/mahmoodlab/CLAM) with **Additive Multiple Instance Learning [(AddMIL)](https://openreview.net/forum?id=5dHQyEcYDgA)** [2]. It was developed with the intention of future extensions of other ABMIL methods, and to support the study:

**Title** 
: Automated detection of cutaneous squamous cell carcinoma (CSCC) in whole slide images of skin biopsies using weakly-supervised learning approaches

**Authors**
: Catherine Chia, Stephan Dooper, Antien Mooyaart, Avital Amir, Marlies Wakkee, and Geert Litjens

## Public datasets
Two internal datasets are used in this study, and both are hosted on an AWS S3 bucket:

`s3://cobra-pathology/`

#### COBRA
The COBRA dataset is publicly accessible via the `bcc` directory:

##### Instruction
1. Install [AWS CLI](https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html)
2. Bucket name: `s3://cobra-pathology/packages/bcc/`
3. To browse: `aws s3 ls --no-sign-request s3://cobra-pathology/packages/bcc/`
4. To download: `aws s3 cp --no-sign-request s3://cobra-pathology/packages/bcc/ <destination_path>`

Relevant directory structure
```
cobra-pathology
└── packages
    ├── bcc
        ├── annotations
        ├── images
    ├── ood
```
#### CSCC 
The CSCC dataset consists of two batches. The batch dated 2016-2020 is accessible via the `ood` directory, whereas the later 2021-2024 batch will be made publicly available during the review process.

##### Instruction
1. Install [AWS CLI](https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html)
2. Bucket name: `s3://cobra-pathology/packages/ood/`
3. To browse: `aws s3 ls --no-sign-request s3://cobra-pathology/packages/ood/`
4. To download: `aws s3 cp --no-sign-request s3://cobra-pathology/packages/ood/ <destination_path>`

Relevant directory structure
```
cobra-pathology
└── packages
    ├── bcc
    ├── ood
        ├── annotations
        ├── images
```

## Citations
If you use this repository, please cite:
```
@software{diagnijmegen_pathology_clam_addmil,
  title={CLAM and Additive MIL},
  author={Catherine Chia, Ivan Slootweg, Stephan Dooper, and Geert Litjens},
  url={https://github.com/DIAGNijmegen/pathology-clam-addmil},
  year={2025}
}
```

and CLAM:
```
@article{lu2021data,
  title={Data-efficient and weakly supervised computational pathology on whole-slide images},
  author={Lu, Ming Y and Williamson, Drew FK and Chen, Tiffany Y and Chen, Richard J and Barbieri, Matteo and Mahmood, Faisal},
  journal={Nature Biomedical Engineering},
  volume={5},
  number={6},
  pages={555--570},
  year={2021},
  publisher={Nature Publishing Group}
}
```


If you use the `CSCC` dataset, please cite:
```
@article{chia2026cscc,
  title={Automated detection of cutaneous squamous cell carcinoma (CSCC) in whole slide images of skin biopsies using weakly-supervised learning approaches},
  author={Chia, Catherine et al.},
  year={2026}
}
```

If you use the `BCC` dataset, please cite:
```
@article{geijs2023bcc,
  title={Detection and subtyping of basal cell carcinoma in whole-slide histopathology using weakly-supervised learning},
  author={Geijs, Daan et al.},
  year={2023}
}
```






