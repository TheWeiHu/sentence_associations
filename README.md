The purpose of this project is to investigate whether ... models have an embedded personality. Building on Dr. Victor Swift's project (https://arxiv.org/abs/2002.10284),
we consider the sentence embeddings ... and.... 

# use 3.5 < python < 3.8

# Running Bert-As-Service Locally

Description of the Repo is found here: https://github.com/hanxiao/bert-as-service.
From there, download the "BERT-Base, Uncased" model under step 1 of the "Getting Started" section. From there, set-up a Python virtual environment with Tensorflow >= 1.10 (avoid using Tensorflow 2; it leads to errors, by running: 

```bert-serving-start -model_dir /Users/weihu/Downloads/uncased_L-12_H-768_A-12 -num_worker=1```

The images produced do not have...

Note: do not upload to GitHub repo, as permission to post dataset.

# TODO: clean up requirements (removing unnecessary)
# TODO: check if Ben can follow
