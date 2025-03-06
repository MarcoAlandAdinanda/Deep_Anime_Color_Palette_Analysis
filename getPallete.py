import pandas as pd
from tqdm import tqdm  
from colorPallete import ColorPaletteExtractor

class DataFramePaletteProcessor:
    def __init__(self, dataframe, url_column='img', num_clusters=5):
        """
        Initialize the processor with a DataFrame and configuration.
        
        Args:
            dataframe: pandas DataFrame containing image URLs
            url_column: name of the column containing URLs
            num_clusters: number of colors to extract per image
        """
        self.df = dataframe.copy()
        self.url_column = url_column
        self.num_clusters = num_clusters
        self.analyzer = ColorPaletteExtractor()
        
    def _process_single_url(self, url):
        """Helper method to process a single URL"""
        try:
            self.analyzer.load_image_from_url(url)
            self.analyzer.extract_palette(num_clusters=self.num_clusters)
            return self.analyzer.get_palette().tolist()
        except Exception as e:
            print(f"Error processing {url}: {str(e)}")
            return None
        
    def add_palette_column(self, column_name='palette', show_progress=True):
        """
        Add a palette column to the DataFrame.
        
        Args:
            column_name: name for the new palette column
            show_progress: whether to display a progress bar
        """
        tqdm.pandas(desc="Extracting color palettes")
        
        if show_progress:
            self.df[column_name] = self.df[self.url_column].progress_apply(
                self._process_single_url
            )
        else:
            self.df[column_name] = self.df[self.url_column].apply(
                self._process_single_url
            )
            
        return self.df

# Example usage
if __name__ == "__main__":
    # Read anime dataset
    df_title = pd.read_csv("filtered_title.csv", delimiter=";")
    filtered_title = df_title["jp_title"].unique().tolist()

    # Read character dataset
    df_character = pd.read_csv("CharacterDataset/merged_char_dataset.csv")

    # Filter characters whose `jp_title` is in `filtered_title`
    print(df_character.shape)
    filtered_character = df_character[df_character["jp_title"].isin(filtered_title)]

    
    # Process DataFrame
    processor = DataFramePaletteProcessor(
        dataframe=filtered_character,
        url_column='img',
        num_clusters=30
    )
    
    # Add palette column with progress tracking
    processed_df = processor.add_palette_column(column_name='colors')
    
    processed_df.to_csv("char_color_palette.csv")

    # Display results
    print("[EXPORTING] char_color_palette.csv")