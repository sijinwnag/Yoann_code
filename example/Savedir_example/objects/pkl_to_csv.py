import pickle as pkl
import pandas as pd
with open(r"C:\Users\sijin wang\Desktop\TOR_dell\literature_review\yoanncode\example\Savedir_example\objects\ltsDF_ID0_2022-01-21-13-50-48.pkl", "rb") as f:
    object = pkl.load(f)

df = pd.DataFrame(object)
df.to_csv(r'ltsDF_ID0_2022-01-21-13-50-48.pkl.csv')
