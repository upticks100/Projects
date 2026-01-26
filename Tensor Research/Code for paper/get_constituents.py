import numpy as np 
import pandas as pd 
from pathlib import Path

_HERE = Path(__file__).resolve().parent

df = pd.read_csv(_HERE / 'sp500_constituents_08-14-25.csv')
list_gvkeys = [x for x in df['gvkey']]

with open(_HERE / "gvkeys.txt", "w") as f:
    for x in list_gvkeys: 
        f.write(str(x) + "\n")

