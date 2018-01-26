# Model Expansion
#### Satchel Grant – Jan 25, 2018

# Description
This project seeks to achieve strengthened training by progressively increasing the parameters used in classification. It also has potential to combat the _catastrophic forgetting problem_. 

2 recurrent networks are created with the goal of word prediction. The two networks have different sized embeddings and early weight matrices. The smaller of the two networks (the _core_ network) is trained on a [4th grade reading level text dataset](https://www.microsoft.com/en-us/research/publication/mctest-challenge-dataset-open-domain-machine-comprehension-text/). 

The embedding and early net parameters of the _core_ network are then copied into subsets of the corresponding Variables of the larger network (the _expanded_ network). The _expanded_ network is then trained on a [dataset](https://s3.amazonaws.com/text-datasets/nietzsche.txt) with a larger, more difficult vocabulary. At the end of each epoch, the subsets of the _expanded_ network are reset to the _core_ network values.

The final training alternates training between the _expanded_ network and _core_ network. The appropriate _core_ network parameters are averaged with their old values and the updated subset parameters from the _expanded_ network training.


# Understanding the Code
`prototype.py` is the main script that handles data preperation and initiates training.
`model.py` is a class that handles the sepration of the two networks.
`RecurrentUnit.py` is a class that wraps the recurrent model (in this case 2 stacked GRUs).
`DoubleGRU.py` is a class that contains 2 GRUs stacked together.
`utils.py` contians a single helper function for parsing the text files

