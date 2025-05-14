import pandas as pd
# load both datasets
fake_df = pd.read_csv('Fake.csv')
true_df = pd.read_csv('True.csv')
# add label columns
fake_df['label'] = 'Fake'
true_df['label'] = 'True'
#combine both into one dataframe
Merged_df = pd.concat([fake_df,true_df])
print(Merged_df['label'].value_counts())
Merged_df .to_csv("news.csv", index=False)
print("data sets merged successfully into 'news.csv'!")
print(Merged_df.head())
print(Merged_df.tail())