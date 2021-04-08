"""
Import tools for obtaining data.
"""
import zipfile
from kaggle.api import KaggleApi
import pandas


def get_data():
    """
    Uses Kaggle API to download astronaut dataset, zipfile to unzip
    the dataset, and then loads it into variable nasa_astronaut_dataset
    using Pandas.

    Note that the Kaggle Dataset was published by NASA as the "Astronaut Fact
    Book" (April 2013 edition) and provides educational /military backgrounds
    for astronauts from 1959-2013. In some cells are blank, indicating special
    circumstances. A missing military rank/branch indicates the person was a
    civilian, a missing death date/mission means the person has not died yet,
    and a blank year/group indicates someone who was a payload specialist, or
    someone who did not undergo formal astronaut selectrion/training and were
    not designated NASA astronauts. They were nomiated by a non-US
    space agency or a paylad sponsor, typically from the research community.

    Returns:
        nasa_astronaut_dataset: A pandas dataframe containing information
        about NASA astronauts.
    """
    api = KaggleApi()
    api.authenticate()
    api.dataset_download_files('nasa/astronaut-yearbook')
    with zipfile.ZipFile("astronaut-yearbook.zip", 'r') as zip_ref:
        zip_ref.extractall()
    nasa_astronaut_dataset = pandas.read_csv("astronauts.csv")
    return nasa_astronaut_dataset
