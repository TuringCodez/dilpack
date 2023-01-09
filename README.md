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
├── example - All sample Juptyer Notebooks can be placed here
│   └── image
│       ├── Sample Demo Jupyter Notebooks related to image processing
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
└── tests
    └── test_mahalanobis.py
```

Developing locally
==================

Run the following command in the package root folder to install the current package in editable mode

```
pip install -e .
```