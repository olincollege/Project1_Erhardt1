import matplotlib.pyplot as plt
import pandas
import requests
import zipfile
import wikipedia
import kaggle
from kaggle.api.kaggle_api_extended import KaggleApi
import itertools
import math
from math import isnan
import numpy as np
import plotly.graph_objects as go

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

def most_space_walks(nasa_astronaut_dataset):
    """
    """
    row = nasa_astronaut_dataset["Space Walks (hr)"].idxmax()
    astronaut = nasa_astronaut_dataset["Name"][row]
    hours = nasa_astronaut_dataset["Space Walks (hr)"][row]

    return (f"{astronaut} has the most hours outside a vehicle in space with {hours} hours.")

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
        split_list = i.split("; ")
        college_list += split_list
    college_frequency = {}
    for college in college_list:
        if college not in college_frequency:
            college_frequency[college] = 1
        else:
            college_frequency[college] += 1
    new ={k: v for k, v in sorted(college_frequency.items(),
                                  key=lambda item: item[1], reverse = True)}
    new.pop("nan", None)
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
    gender = ["Female", "Male"]
    count_row = nasa_astronaut_dataset.shape[0]
    gender = ["Female", "Male"]
    count_row = nasa_astronaut_dataset.shape[0]
    plt.style.use("ggplot")
    plt.bar(gender, gender_occurrence_name, width=0.8, label='Civilian', color='silver')
    plt.bar(gender, gender_military, width = 0.8, label = 'Military', color = 'gold')
    plt.xlabel("Gender")
    plt.ylabel("Number of Astronauts")
    plt.title("Gender Distribution of Astronauts")

    plt.legend(loc="upper left")
    plt.show()
    pass


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

def average_age_vs_group(nasa_astronaut_dataset):
    average_age = {}
    for i in range(19):
        grouper = nasa_astronaut_dataset[nasa_astronaut_dataset.Group == i]
        
        average_age[i] = average(grouper, "Selection Age")
        
    average_age.pop(0, None)
    
    group = list(average_age.keys())
    
    age = list(average_age.values())
    
    x_ticks = np.arange(min(group), max(group), 1)
    plt.xticks(x_ticks)
    plt.scatter(group, age)
    
    pass

def most_common_state(nasa_astronaut_dataset):
    """
    """
    frequent_state = frequency(nasa_astronaut_dataset, 19)
    top_ten_state = list(frequent_state)[:10]
    all_frequency = list(frequent_state.values())
    top_ten_frequency = all_frequency[:10]
    top_ten = dict(list(frequent_state.items())[0: 10]) 

    plt.style.use("ggplot")
    plt.bar(top_ten_state, top_ten_frequency, width=0.8, color='pink')
    plt.xlabel("State")
    plt.ylabel("Number of Astronauts")
    plt.title("Most Common Astronaut Home States")

    plt.show()

    pass

def violin_plot(nasa_astronaut_dataset):

  
    # Creating 3 empty lists
    for i in range(19):
        str(i) = []
        
    pass
    
"""
    # Filling the lists with random value
    for z in range

    for i in range(100):
        n = randint(1, 100)
        l2.append(n)

    for i in range(100):
        n = randint(1, 100)
        l3.append(n)

    random_collection = [l1, l2, l3]

    # Create a figure instance
    fig = plt.figure()

    # Create an axes instance
    ax = fig.gca()

    # Create the violinplot
    violinplot = ax.violinplot(random_collection)
    plt.show()
    """