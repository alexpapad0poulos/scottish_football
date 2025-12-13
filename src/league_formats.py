import pandas as pd

def apply_split(table_df, top_n=6):
    """
    Split table into top and bottom groups after Phase 1.
    Returns two lists of teams.
    """
    top = table_df.iloc[:top_n].index.tolist()
    bottom = table_df.iloc[top_n:].index.tolist()
    return top, bottom