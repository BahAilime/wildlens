import pandas as pd
import re

def process_species_regions(csv_path):
    df = pd.read_csv(csv_path, sep=";", encoding="utf-8")
    
    # Extraire et normaliser les régions
    regions_list = df["Région"].dropna().unique()
    split_regions = set()
    for regions in regions_list:
        split_regions.update(re.split(r"[(),]| et ", regions))
    split_regions = {region.strip() for region in split_regions if region.strip()}
    
    regions = list(split_regions)
    
    def detect_regions(region_str):
        found_regions = set()
        if pd.notna(region_str):
            for region in re.split(r"[(),]| et ", region_str):
                region = region.strip()
                if region in regions:
                    found_regions.add(region)
        return found_regions
    
    df["Regions"] = df["Région"].apply(detect_regions)
    
    for region in regions:
        df[region] = df["Regions"].apply(lambda x: 1 if region in x else 0)
    
    df.drop(columns=["Région"], inplace=True)
    
    return df

csv_path = "infos_especes.csv"
df_result = process_species_regions(csv_path)
print(df_result.head())

df_result.to_csv("infos_especes_lieu.csv")