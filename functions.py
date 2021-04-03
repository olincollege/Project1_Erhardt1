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
from pylab import plot, title, xlabel, ylabel, savefig, legend, array

def get_image(url):
    response = requests.get(url)
    file = open("class21.png", "wb")
    file.write(response.content)
    file.close()
    return file

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
    nasa_astronaut_dataset = nasa_astronaut_dataset.rename({"Birth Date": "Birth Year"}, axis=1, inplace = True)
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

def highest(nasa_astronaut_dataset, column):
    """
    """
    row = nasa_astronaut_dataset[column].idxmax()
    astronaut = nasa_astronaut_dataset["Name"][row]
    hours = nasa_astronaut_dataset[column][row]

    return (f"{astronaut} has the most {column} with a total of {hours} hours.")


def filter_by_year(nasa_astronaut_dataset, year_min, year_max):
    """
    """
    below = nasa_astronaut_dataset[nasa_astronaut_dataset.Year <= year_max]
    above = below[below.Year >= year_min]
    return above

def filter_by_group(nasa_astronaut_dataset, group_min, group_max):
    """
    """
    below = nasa_astronaut_dataset[nasa_astronaut_dataset.Group <= group_max]
    above = below[below.Group >= group_min]
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

def engineer(nasa_astronaut_dataset):
    new = frequency(nasa_astronaut_dataset, 8)

    majors = new.keys()

    engineer = 0
    non = 0
    for major in majors:
        if "Engineering" in major:
            engineer += 1
            continue
        non +=1
    total = engineer + non
    print(f"{engineer / total * 100} % of Astronauts majored in some kind of Engineering")


def gender_military(nasa_astronaut_dataset):
    """
    """
    gender_occurrence = nasa_astronaut_dataset.groupby('Gender').count()
    gender_military = gender_occurrence["Military Rank"]
    gender_occurrence_name = gender_occurrence["Name"]
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
    plt.plot(years,astronauts)
    plt.xlabel("Year")
    plt.ylabel("Number of Astronauts")
    plt.title("Number of NASA Astronauts over the Years")
    pass


def grad_school_vs_not_grad_school(nasa_astronaut_dataset):
    """
    """
    gradschool = 0
    not_gradschool = 0
    grad_or_not = list(nasa_astronaut_dataset.iloc[:,9])
    for i in grad_or_not:
        if type(i) == str:
            gradschool +=1
            continue
        not_gradschool += 1
        
    return [gradschool, not_gradschool]
        

def age_vs_group(nasa_astronaut_dataset):
    
    values = []
    for i in range(19):
        grouper = nasa_astronaut_dataset[nasa_astronaut_dataset.Group == i]
        
        youngest = (grouper["Selection Age"]).min()
        mean = (grouper["Selection Age"]).mean()
        oldest = (grouper["Selection Age"]).max()
        
        values.append((youngest, mean, oldest))
    groups = array(list(range(19)))
   
    for temp in zip(*values):
        plot(groups, array(temp))
    
    title('Age Distribution of Astronaut Groups')
    xlabel('Group Number')
    ylabel('Age (yrs)')
    legend(['min', 'avg', 'max'], loc='upper right')
    
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
    
def female_astronauts_decade(nasa_astronaut_dataset):
    """
    """
    counter = 1
    start = 1949
    end = 1960
    frequency_per_decade = []

    while counter <= 6:
        decade = filter_by_year(nasa_astronaut_dataset, start, end)
        gender_frequency = frequency(decade, 6)
        values = list(gender_frequency.values())
        if len(values) == 2:
            percentage_women = values[1] / sum(values) * 100
            frequency_per_decade.append(percentage_women)
        else:
            frequency_per_decade.append(0)
        start += 10
        end += 10
        counter += 1
    
    decade = [1950, 1960, 1970, 1980, 1990, 2000]
    plt.step(decade, frequency_per_decade, color = 'blue')
    plt.xlabel('Decade')
    plt.ylabel('% Female Astronauts (pct.)')
    plt.title('Female Astronauts Selected Per Decade')

    plt.show()

def military_college_over_time(nasa_astronaut_dataset):
    values = []
    for i in range(23):
        grouper = nasa_astronaut_dataset[nasa_astronaut_dataset.Group == i]
        
        college_list = []
        colleges = grouper["Alma Mater"]
        for i in colleges:
            i = str(i)
            split_list = i.split("; ")
            college_list += split_list
            
        military_prep = 0
        non = 0
        for i in college_list:
            if "US " not in i and "The Citadel" not in i:
                military_prep += 1
            else:
                non += 1
        if (military_prep + non) > 0:
            values.append(military_prep / (military_prep + non)* 100)
            continue
        values.append(0)
    year = range(23)
    plt.scatter(year, values)
    plt.ylim(5,110)
    plt.xlabel('Group Number')
    plt.ylabel('% From a Military College(pct.)')
    plt.title('Percentage of Astronauts from a Military College')

    plt.show()


def grad_school_over_time(nasa_astronaut_dataset):
    
    sixties = filter_by_year(nasa_astronaut_dataset, 1960, 1970)
    inthesix = grad_school_vs_not_grad_school(sixties)
    
    seventies = filter_by_year(nasa_astronaut_dataset, 1970,1980)
    intheseven = grad_school_vs_not_grad_school(seventies)
    
    eighties = filter_by_year(nasa_astronaut_dataset, 1980, 1990)
    intheeight = grad_school_vs_not_grad_school(eighties)
    
    ninties = filter_by_year(nasa_astronaut_dataset, 1990, 2014)
    inthenine = grad_school_vs_not_grad_school(ninties)
    
    labels ='Gradschool','Not'
    
    fig, axs = plt.subplots(2, 2)
    axs[0, 0].pie(inthesix, labels=labels,autopct='%1.1f%%')
    axs[0, 0].set_title('1960s')
    axs[0, 1].pie(intheseven, labels=labels, autopct='%1.1f%%')
    axs[0, 1].set_title('1970s')
    axs[1, 0].pie(intheeight, labels=labels, autopct='%1.1f%%')
    axs[1, 0].set_title('1980s')
    axs[1, 1].pie(inthenine, labels=labels, autopct='%1.1f%%')
    axs[1, 1].set_title('90s and 2000s')
    
    fig.suptitle('Astronaut Post-Grad Trends through the Decades',x=.5,y=1, fontsize=13)

    # Hide x labels and tick labels for top plots and y ticks for right plots.
    for ax in axs.flat:
        ax.label_outer()
    ttl = ax.title
    ttl.set_position([.5, 1.5])
        
def first_v_last(nasa_astronaut_dataset):
    first_three = filter_by_group(nasa_astronaut_dataset, 1, 1)
    
    gender_occurrence = first_three.groupby('Gender').count()
    gender_military = gender_occurrence["Military Rank"]
    gender_occurrence_name = gender_occurrence["Name"]
    gender = ["Male"]
    
    count_row = first_three.shape[0]
    plt.style.use("ggplot")
    plt.bar(gender, gender_occurrence_name, width=0.8, label='Civilian', color='silver')
    plt.bar(gender, gender_military, width = 0.8, label = 'Military', color = 'gold')
    plt.xlabel("Gender")
    plt.ylabel("Number of Astronauts")
    plt.title("Gender Distribution of Astronauts in First Astronaut Class")

    plt.legend(loc="upper left")
    plt.show()
    
    last_three = filter_by_group(nasa_astronaut_dataset, 20, 20)
    gender_occurrence = last_three.groupby('Gender').count()
    gender_military = gender_occurrence["Military Rank"]
    gender_occurrence_name = gender_occurrence["Name"]
    gender = ["Female", "Male"]
    
    count_row = last_three.shape[0]
    plt.style.use("ggplot")
    plt.bar(gender, gender_occurrence_name, width=0.8, label='Civilian', color='silver')
    plt.bar(gender, gender_military, width = 0.8, label = 'Military', color = 'gold')
    plt.xlabel("Gender")
    plt.ylabel("Number of Astronauts")
    plt.title("Gender Distribution of Astronauts in Last Astronaut Class")

    plt.legend(loc="upper left")
    plt.show()
    pass
