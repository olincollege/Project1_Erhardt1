import matplotlib.pyplot as plt
import pandas
import requests
import zipfile
import wikipedia
import kaggle
from kaggle.api.kaggle_api_extended import KaggleApi
import itertools
import math
import numpy as np
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

def change_dates(nasa_astronaut_dataset):
    """
    """
    nasa_astronaut_dataset['Birth Date'] = nasa_astronaut_dataset['Birth Date'].str[-4:]
    nasa_astronaut_dataset["Birth Date"] = nasa_astronaut_dataset["Birth Date"].astype(float)
    nasa_astronaut_dataset["Selection Age"] = (nasa_astronaut_dataset["Year"]
                                               - nasa_astronaut_dataset["Birth Date"])
    return nasa_astronaut_dataset

def add_selection_age(nasa_astronaut_dataset):
    """
    """
    nasa_astronaut_dataset["Birth Date"] = nasa_astronaut_dataset["Birth Date"].astype(float)
    return nasa_astronaut_dataset

def add_birth_state(nasa_astronaut_dataset):
    """
    Creates a new column with the home state of each astronaut.
    
    args:
        nasa_astronaut_dataset: the df being manipulated.
        
    return:
        nasa_astronaut_dataset: the updated df. 
    """
    nasa_astronaut_dataset["Birt State"] = nasa_astronaut_dataset["Birth Place"].str[-2:]
    return nasa_astronaut_dataset

def highest_flight_hours(nasa_astronaut_dataset):
    """
    """
    row = nasa_astronaut_dataset["Space Flight (hr)"].idxmax()
    astronaut = nasa_astronaut_dataset["Name"][row]
    hours = nasa_astronaut_dataset["Space Flight (hr)"][row]

    return (f"{astronaut} has the most flight hours with a total of {hours} hours.")

def filter_by_year(nasa_astronaut_dataset, year_min, year_max):
    """
    """
    below = nasa_astronaut_dataset[nasa_astronaut_dataset.Year < year_max]
    above = below[below.Year > year_min]
    return above

def filter_by_group(nasa_astronaut_dataset, group_min, group_max):
    """
    """
    below = nasa_astronaut_dataset[nasa_astronaut_dataset.Group < group_max]
    above = below[below.Group > group_min]
    return above

def frequency(nasa_astronaut_dataset,column):
    """
    """
    colleges = nasa_astronaut_dataset.iloc[:,column]
    college_list = []
    for i in colleges:
        i = str(i)
        split_list = i.split(";")
        college_list += split_list
    college_frequency = {}
    for college in college_list:
        if college not in college_frequency:
            college_frequency[college] = 1
        else:
            college_frequency[college] += 1
    new ={k: v for k, v in sorted(college_frequency.items(),
                                  key=lambda item: item[1], reverse = True)}
    return new

def tops(new, top_number):
    """
    """
    return dict(itertools.islice(new.items(), top_number))

def gender_military(nasa_astronaut_dataset):
    """
    """
    gender_occurrence = nasa_astronaut_dataset.groupby('Gender').count()
    gender_military = gender_occurrence["Military Rank"]
    gender_occurrence_name = gender_occurrence["Name"]
    return gender_occurrence, gender_occurrence_name

def most_space_walks(nasa_astronaut_dataset):
    """
    """
    row = nasa_astronaut_dataset.nlargest(3, "Space Walks")
    row.sort_values(["Space Walks"], ascending=True)
    df1 = row[["Name", "Space Walks"]]
    return df1

def average(nasa_astronaut_dataset, column):
    """
    """
    return nasa_astronaut_dataset[column].mean()


def plot_astronauts_vs_time(nasa_astronaut_dataset):
    astronauts_per_year = frequency(nasa_astronaut_dataset, 1)
    new ={k: v for k, v in sorted(astronauts_per_year.items(),
                                  key=lambda item: item[0], reverse = False)}
    years = list(new.keys())

    astronauts = list(new.values())

    years.pop()
    astronauts.pop()
    
    years = list(map(float, years))
    astronauts = list(map(float, astronauts))

    x_ticks = np.arange(min(years), max(years), 5)
    plt.xticks(x_ticks)
    plt.plot(years, astronauts)
    pass


def grad_school_vs_not_grad_school(nasa_astronaut_dataset):
    """
    """
    gradschool = 0
    not_gradschool = 1
    grad_or_not = list(nasa_astronaut_dataset.iloc[:,9])
    for i in grad_or_not:
        if type(i) == str:
            gradschool +=1
            continue
        not_gradschool += 1
        
    labels = "Grad School", "Not Grad School"
    sizes = [gradschool, not_gradschool]
    fig1, ax1 = plt.subplots()
    ax1.pie(sizes, labels=labels, autopct='%1.1f%%',
            shadow=True, startangle=90)
    ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

    plt.show()

