# Project Objective

In this project, I propose an approach to test for biases in pretrained language models used in NLP. Extending on Dr. Victor Swift's [research on word embeddings](https://arxiv.org/abs/2002.10284), and using [sentence completion survey data](http://openpsychometrics.org/_rawdata/SENTANCES1.zip), I set out to examine whether these models "associate" more strongly with certain personality types.

# Methodology

Three assumptions are being made in this project:
- First, I assume sentence completion tests are meaningful for personality analysis.
    - Sentence completion tests typically provide respondents with beginnings of sentences, referred to as "stems", and respondents then complete the sentences in ways that are meaningful to them [[source](https://en.wikipedia.org/wiki/Sentence_completion_tests)].
    - Modern tests, built on Carl Jung's word association test, have become more popular in recent decades. 
    - At one point, it was the [13th most-used psychological instrument](https://doi.org/10.1080/0091651X.1965.10120175).
- Further, I assume the [cosine similarity](https://en.wikipedia.org/wiki/Cosine_similarity) of two sentence embeddings is a good measure for how closely the model associates the two concepts.
    - For example, if the embedding for the stem "One should never trust..." is similar to that of "pirates," it would suggest the model considers "pirates" to be untrustworthy.
- Lastly, I assume a modified form of this test, in which respondants describe their inclinations towards other respondents' completions rather than providing their own, has similar validity.

In the project, sentence embeddings are generated using:
- the BERT Base model, through the ```Bert-As-Service``` client.
- the Distil Roberta Base model, through UKPLab's ```sentence-transformers``` package.

From there, we calculate the cosine similarity between the embeddings for each stem and its corresponding respondent completions. Since, we have big five personality score for each respondent, we can apply statistical methods to examine whether the model prefers respondents with certain types of personality.

For example, if the model always gives high scores to responses from extroverted people and low ones to responses from neurotic people, it would be suggestive of a bias.

# Development

I did not include the original dataset in this repository, since I have not gotten the authors' permissions. Download the original dataset from [HERE](http://openpsychometrics.org/_rawdata/SENTANCES1.zip), and unzip it in the root directory.

Be sure to:
- use a ```Python``` version between ```3.5``` and ```3.8```, which allows us to:
- install a Tensorflow Version ```1.x```, for some ```x > 10```. 
    - Tensorflow ```2``` or Tensorflow < ```1.10``` does not allow us to run ```Bert-As-Service```.
    - Alternatively, use the ```requirements.txt``` file to install all the dependencies.

From, there, launch the ```Bert-As-Service``` client:
- download the "BERT-Base, Uncased" model from the source [README](https://github.com/UKPLab/sentence-transformers) under *Step 1* of the *Getting Started* section, and unzip it.
- in a new terminal window, run ```bert-serving-start -model_dir <path to the unzipped model> -num_worker=1```, which starts the client.

From there, running ```main.py``` will calculate cosine similarities scores using each of the models, and save the results in the ```output``` folder in a csv file named after the model.

The ```analysis.ipynb``` Jupyter notebook shows my approach for statistically analyzing the similarity scores. Change the ```MODEL_NAME``` field to apply the analysis to similarity scores calculated for a different model.

# Conclusions

There is not enough evidence in my analysis to conclude a bias towards certain personality types. The results would be consistent with what would be obtained if the model had a completely neutral personality (50th-percentile on each trait).

There is weak evidence suggesting the model disfavors responses from respondants with extreme scores on any of the big fives (e.g. really high or really low scores for openness).

To further explore, it would be worthwhile to consider: 
- using longer tests
    - these association tests usually have 40 to 100 stems (rather than a mere 6, as in our dataset)
    - 6 reponses cannot contain all that much information about one's personality. 
    - longer tests give us stronger statistical power
- examining a greater variety of NLP models
    - RNN-based models, Doc2Vec, InferSent
    - other flavors of Transformers
- looking into [other approaches for extracting embeddings](https://mccormickml.com/2019/05/14/BERT-word-embeddings-tutorial/)
    - for example, ```Bert-As-Service``` just uses the output from the second last layer of the BERT model, which is arbitrary (though, I've tested to make sure the embedding is fairly reasonble, e.g. queen - female + male ~ king)
    - alternatively, we could have have: concatenated the last four layers, summed the last three layers, etc.

In any case, this pilot project introdduces a feasible method for identifying models with strong personalities preferences.