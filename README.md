Everything that is on here is for public use, and I encourage everyone who is interested to download, clone, edit, use and
experiment with these models at their hearts' desire! Also, feel free to let me know if there a re any bugs or improvements you
think would be great to have, and I might just get around and update those, too.
As far as dependencies go, I have tried to make it as self-contained as possible. You will need, depending on the model, some of the following packages/moduels installled:

- nltk
- networkx
- and whatever else it says at the top of the scripts.

A bit about the model on here:

TextRank

The TextRank_LOTR folder contains a from scratch implementation of the 2003 paper by Rada Mihalcea and Paul Tarau. It scans across a text and extracts keyphrases based on the co-occurrence within the text of the words forming the keyphrases. As the paper states, it is based on the original PageRank algorithm used for website ranking in the early days of the internet by google. It contains the functions doing the heavy lifting as well as a run-and-show script that applies the TextRank model on a given input string.

The files:
- "LOTR_1", "LOTR_2" and "LOTR_3":  Textfiles containing the first, second and third part of the "The Lord of the rings" trilogy.
- "stopwords.txt":                  Textfiles containing stopwords
- "TextRank_2003_paper":            The original paper outlining the TextRank algorithm. The implementation is exclusively based
                                    on this.
- "TextRank.py":                    Contains the functions used to run the algorithm.
- "TextRank_running_script.py":     Script used for running the TextRank algorithm (LOTR_3 is chosen by default). Usage is
                                    "python -m TextRank_running_script" from within the 'TextRank_LOTR' folder. Type
                                    "python -m TextRank_running_script -h" for information on model/script parameters.
