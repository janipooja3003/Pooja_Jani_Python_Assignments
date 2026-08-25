from pathlib import Path
import pandas as pd
import streamlit as st

from utils.data_loader import load_data 
#from .filters import sidebar_filters, apply_filters


def load_application_train_data() -> pd.DataFrame:
    csv_path = Path(__file__).resolve().parents[1] / "data" / "application_train.csv"
    return load_data(csv_path)
