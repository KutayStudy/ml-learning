import pytest
import numpy as np

def is_even(x):
    return x % 2 == 0

@pytest.mark.parametrize(
    "x,expected",
    [ (4,True),
      (0,True),
      (7,False)])

def test_is_even(x,expected):
    assert is_even(x) == expected

@pytest.fixture
def numbers():
    return [1,2,3,4,5]

def test_sum(numbers):
    assert sum(numbers) == 15

def test_length(numbers):
    assert len(numbers) == 5

def test_write_file(tmp_path):
    file_path = tmp_path / "example.txt"
    file_path.write_text("hello")
    assert file_path.read_text() == "hello"

def test_seed_reproducibility():
    rng1 = np.random.default_rng(42)
    rng2 = np.random.default_rng(42)
    values1 = rng1.random(5)
    values2 = rng2.random(5)
    assert np.array_equal(values1,values2)