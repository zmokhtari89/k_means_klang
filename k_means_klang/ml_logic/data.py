import pandas as pd
from pathlib import Path
from colorama import Fore, Style


def load_data(
    cache_path: Path,
    data_has_header=True):
    """
    loads the dataset
    """
    if cache_path.is_file():
        print(Fore.BLUE + "\nLoad data from local CSV..." + Style.RESET_ALL)
        df = pd.read_csv(cache_path, header='infer' if data_has_header else None)
    print(f"✅ Loading Completed")
    return df
