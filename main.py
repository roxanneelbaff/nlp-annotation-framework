

import os

from textmining_utility import lexicons

print(os.path.realpath(os.path.join(os.path.dirname(__file__), '.')))
print(os.path.realpath(os.path.join(os.path.dirname(__file__), '..')))
print(os.path.realpath(os.path.join(os.path.dirname(__file__), '...')))
import pandas as pd
lexicons.count_nrc_emotions_and_sentiments(pd.DataFrame([{"text": "This sucks and is horrible"},
                                                         {"text": "This is GREAT"}]))