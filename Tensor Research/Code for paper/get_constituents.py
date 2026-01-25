import numpy as np 
import pandas as pd 

df = pd.read_csv('/student/mcnama53/VScode/Alpha/Research/Code for paper/sp500_constituents_08-14-25.csv')
list_gvkeys = [x for x in df['gvkey']]

with open("/student/mcnama53/VScode/Alpha/Research/Code for paper/gvkeys.txt", "w") as f:
    for x in list_gvkeys: 
        f.write(str(x) + "\n")

