# LSTM

This repository contains the code for a one-layer LSTM network with 1024 hidden dimensions that, when trained on a text such as a book, can generate similar text. 

## Learning

Takes in a sequence of characters (e.g. a book), and creates a probability distribution for the next character given the last e.g. thirty characters.

## Generation

Text is generated by randomly setting the first character of the sentence and then always picking the token with the highest probability predicted by your model. In order to 'give a fair chance' to overall unlikely characters, such as punctuation, a temperature parameter is implemented.

## Additional Details

* Teacher Forcing implemented, to increase efficiency of learning by not using the (possibly faulty) intermediate outputs of the RNN as inputs, but instead using the ground truth.
* Temperature parameter for sampling implemented, in order to increase the likelihood of punctuation

## Sample Generated Text
When trained on Grimm's _Märchen_ from https://www.gutenberg.org:

* On the third day the queen did not go into the garden, but waited inside her chamber
* King Grisly-beard, who had been disguised
* \nRelease Date: April, 2001\n
