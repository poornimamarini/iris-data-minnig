import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv('iris.csv')
for keys in data.keys() :
    print(keys)