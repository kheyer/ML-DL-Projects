# Text Classification with RNNs

This project builds a text classification model based on the [Blog Authorship Corpus](http://u.cs.biu.ac.il/~koppel/BlogCorpus.htm).
First the corpus is used to create a language model. Then the learned weights from the language model are used to create a classification 
model. The goal of the classification model will be to determine the gender of the author of a blog post.

In the first attempt (Blog Classification RNN), I try to train the language model from scratch. It's a difficult process that doesn't really go anywhere. The resulting classification model doesn't work at all, achieving only 50% accuracy.

In the second attempt (Blog Classification V2), I use a pretrained model to jump start the language model before training on the blogs corpus. The resulting classifier achieves 77% accuracy. Not world class, but definitely an improvement.
