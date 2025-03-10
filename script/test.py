import pandas as pd

# Read anime dataset
df_title = pd.read_csv("filtered_title.csv", delimiter=";")
filtered_title = df_title["jp_title"].unique().tolist()

# Read character dataset
df_character = pd.read_csv("CharacterDataset/merged_char_dataset.csv")

# Filter characters whose `jp_title` is in `filtered_title`
print(df_character.shape)
filtered_character = df_character[df_character["jp_title"].isin(filtered_title)]

print(filtered_character.shape)
# filtered_character.to_csv("filtered_character.csv")