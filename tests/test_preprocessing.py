
# Generated by CodiumAI
from preprocessing import read_in_raw_trade_matrix
from turtle import pd

import pytest


class TestReadInRawTradeMatrix:
    # Test that the function returns a pandas dataframe.
    def test_returns_dataframe(self):
        result = read_in_raw_trade_matrix()
        assert isinstance(result, pd.DataFrame)

    # Function returns a pandas dataframe with an invalid file path
    def test_returns_dataframe_with_invalid_file_path(self):
        with pytest.raises(FileNotFoundError):
            read_in_raw_trade_matrix(file_path="invalid_path")


