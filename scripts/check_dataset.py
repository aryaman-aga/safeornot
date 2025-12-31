import pandas as pd

df = pd.read_csv('data/dataset.csv')

print("Checking for 'This place is not safe' in dataset...")
matches = df[df['text'].str.contains("This place is not safe", case=False)]
print(matches[['text', 'label']].head(10))

print("\nChecking for 'I am being followed' in dataset...")
matches = df[df['text'].str.contains("I am being followed", case=False)]
print(matches[['text', 'label']].head(10))
