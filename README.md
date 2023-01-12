# DilPack

UP Diliman Linear Algebra and Numerical Methods Package.

Linear Algebra and Numerical Methods library by students of AI 211

[Github-flavored Markdown](https://guides.github.com/features/mastering-markdown/)
to write your content.

Directory Structure
-------------------

```
.
├── LICENSE
├── README.md
├── example - All Juptyer Notebooks that demonstrate how the algorithms can be used can be placed here.
│   ├── image
│   │    ├── Sample Demo Jupyter Notebooks related to image processing like Stegonography
|   ├── classifier - Sample Datasets for classification
├── src
│   └── dilpack
│       ├── Main folder
│       ├── classifier
│       │   ├── Algorithms related to classifiers
│       ├── clustering
│       │   ├── Clsutering Algorithms like k-means etc.
│       ├── graph
│       │   └── Algorithms related to graphs
│       ├── image
│       │   ├── Algorithms related to image processing
│       ├── learn
│       │   ├── face
│       │   │   └── face recognition
│       │   └── optimizer
│       │       └── Optimizers like ADAM and gradient descent
│       ├── recommender
│       │   └── Algorithms for recommenders
│       ├── reduction
│       │   └── Algorithms for dimensionality reduction (e.g. PCA)
│       └── utils
│           └── Utility Functions
└── tests - Intended for python unit test
    └── test_mahalanobis.py
```

Developing locally
==================

Run the following command in the package root folder to install the current package in editable mode

```
pip install -e .
```

How to use
==========

The example folder should contain notebooks on how to use the various algorithms in this package as well as
sample datasets


Credits
=======

Folder structure, organization and code herein are contributions of the Turing Batch AI 211 students