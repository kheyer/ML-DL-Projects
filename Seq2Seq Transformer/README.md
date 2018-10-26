# Seq2Seq Transformer

This project was motivated by [fast.ai Lesson 11](http://course.fast.ai/lessons/lesson11.html) and a desire to understand the 
Transformer from [Attention Is All You Need](https://arxiv.org/pdf/1706.03762.pdf). fast.ai lesson 11 builds a seq2seq transformer for 
French to English translation using an LSTM based encoder/decoder model initialized with pretrained word vectors. 
I wanted to apply the same use of pretrained word vectors to a Transformer and expand the dataset used to train the model.

The data used comes from a [corpus](http://www.statmt.org/wmt15/translation-task.html) created by Chris Callison-Burch. 
The dataset was created by crawling english and french versions of web sites to create a parallel corpus. 
This notebook uses the first 2 million sentence pairs from the dataset to train the model.
