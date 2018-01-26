from utils import split_token
from nltk.tokenize import word_tokenize

test_string1 = "\\newline\\newlineYours"
test_string2 = "Yours\\newline\\newline"

splt1 = split_token(test_string1)
for s in splt1:
    print(s)
splt2 = split_token(test_string2)
print(splt2)
