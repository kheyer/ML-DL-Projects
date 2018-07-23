## Predicting with RNNs in FastAI

The current standard method for predicting with an RNN in FastAI is to use `torch.topk` to choose the best output after predicting 
with the model. This lead to some models outputting the same sentence of groups of sentences over and over again. The solution is to 
instead use `torch.multinomial` to select predictions to add some variability into the results.

This notebook shows how to do this with character level and word level FastAI RNN models created by calling `.get_model` on a 
`LanguageModelData ` object.
