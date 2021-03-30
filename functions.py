import matplotlib.pyplot as plt
import pandas
import requests
import zipfile
import wikipedia
import kaggle
from kaggle.api.kaggle_api_extended import KaggleApi

def get_data():
    """
    """
    api = KaggleApi()
    api.authenticate()
    api.dataset_download_files('nasa/astronaut-yearbook')
    with zipfile.ZipFile("astronaut-yearbook.zip", 'r') as zip_ref:
        zip_ref.extractall()
    nasa_astronaut_dataset = pandas.read_csv("astronauts.csv")
    return nasa_astronaut_dataset


def highest_flight_hours():
    """
    """
    row = nasa_astronaut_dataset["Space Flight (hr)"].idxmax()
    astronaut = nasa_astronaut_dataset["Name"][row]
    hours = nasa_astronaut_dataset["Space Flight (hr)"][row]

    return (f"{astronaut} has the most flight hours with a total of {hours} hours.")

