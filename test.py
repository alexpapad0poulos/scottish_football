from src.load_data import load_matches
from src.table import compute_table
from src.league_formats import apply_split

if __name__ == '__main__':
    # Replace with your actual CSV path
    csv_path = './data/scotland_201617.csv'


    matches = load_matches(csv_path)


    # FULL SEASON TABLE (baseline check)
    full_table = compute_table(matches)
    print("=== Final Table (All Matches) ===")
    print(full_table[['Rank', 'Pts', 'GD', 'GF']])


    # PRE-SPLIT TABLE (first 33 matches per team)
    # Assumes matches are ordered chronologically
    matches_pre_split = matches.iloc[: len(matches) * 33 // 38]
    pre_split_table = compute_table(matches_pre_split)


    print("\n=== Pre-Split Table ===")
    print(pre_split_table[['Rank', 'Pts', 'GD', 'GF']])


    top, bottom = apply_split(pre_split_table)
    print("\nTop 6:", top)
    print("Bottom 6:", bottom)