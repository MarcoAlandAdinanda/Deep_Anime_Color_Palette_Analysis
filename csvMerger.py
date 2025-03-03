import pandas as pd

base_path = 'CharacterDataset'

years = [2021, 2022, 2023, 2024]
seasons = ['winter', 'spring', 'summer', 'fall']
contents = ['char']#['anime', 'char'] 

# List to collect all DataFrames
dfs_anime = []
dfs_char = []

for year in years:
    for season in seasons:
        for content in contents:
            path = f"{base_path}/{year}/{content}_{season}_{year}.csv"
            print(f"Reading: {path}")
            df = pd.read_csv(path)
            if content == 'anime':
                dfs_anime.append(df)
            else:
                dfs_char.append(df)

# Concatenate all DataFrames, aligning columns by name
# merged_df_anime = pd.concat(dfs_anime, ignore_index=True)
merged_df_char = pd.concat(dfs_char, ignore_index=True)

# print(f"[EXPORTING] Merged Anime Dataset")
# merged_df_anime.to_csv("merged_anime_dataset.csv")

print(f"[EXPORTING] Merged Character Dataset")
merged_df_char.to_csv("merged_char_dataset.csv")