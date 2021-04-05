import matplotlib.pyplot as plt
import pandas
import requests
import zipfile
import kaggle
from kaggle.api.kaggle_api_extended import KaggleApi
import itertools
import math
import numpy as np
from pylab import plot, title, xlabel, ylabel, savefig, legend, array

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

def change_dates(nasa_astronaut_dataset):
    """
    Takes NASA astronaut dataframe adds each astronauts selection age to a column
    based on their birth date and selection year.

    Args:
        nasa_astronaut_dataset: A pandas dataframe containing information
        from the 2013 NASA Astronaut Factbook.
        
    Returns:
        nasa_astronaut_dataset: Pandas Dataframe with added "Selection Age" column.
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
    
    Args:
        nasa_astronaut_dataset: A pandas dataframe containing information
        from the 2013 NASA Astronaut Factbook.
        
    Returns
        nasa_astronaut_dataset: A revised Pandas dataframe with a
        column containing the birth state initials instead of the
        birth place.
    """
    nasa_astronaut_dataset["Birt State"] = nasa_astronaut_dataset["Birth Place"].str[-2:]
    return nasa_astronaut_dataset

def highest(nasa_astronaut_dataset, column):
    """
    Shows which astronaut has the highest value in a column.

    Args:
        nasa_astronaut_dataset: A pandas dataframe containing information
        from the 2013 NASA Astronaut Factbook.
        
        column: a string representing the name of the Pandas column
        to find the max value of.

    Returns:
        An F-string saying which astronaut has the highest value in the
        column and what the highest value is.
    """
    row = nasa_astronaut_dataset[column].idxmax()
    astronaut = nasa_astronaut_dataset["Name"][row]
    hours = nasa_astronaut_dataset[column][row]

    return (f"{astronaut} has the most {column} with a total of {hours} hours.")


def filter_by_year(nasa_astronaut_dataset, year_min, year_max):
    """
    Filters nasa_astronaut dataset by an input time frame.

    Excludes people not listed in an official astronaut group/year ex. a
    payload specialist in any filter because it is based on the "Year"
    column of the NASA astronaut dataset.

    Args:
        nasa_astronaut_dataset: A pandas dataframe containing information
        from the 2013 NASA Astronaut Factbook.

        year_min: An integar representing the year cutoff from below.

        year_max: An integar representing the year cutoff from above.

    Returns:
        above: A filtered Pandas dataframe with only astronauts within the
        year window.
    """
    below = nasa_astronaut_dataset[nasa_astronaut_dataset.Year <= year_max]
    above = below[below.Year >= year_min]
    return above

def filter_by_group(nasa_astronaut_dataset, group_min, group_max):
    """
    Filters nasa_astronaut dataset by an input selection group frame.

    Excludes people not listed in an official astronaut group ex. a
    payload specialist in any filter because it is based on the "Group"
    column of the NASA astronaut dataset.

    Args:
        nasa_astronaut_dataset: A pandas dataframe containing information
        from the 2013 NASA Astronaut Factbook.

        group_min: An integar representing the year cutoff from below.

        group_max: An integar representing the year cutoff from above.

    Returns:
        above: A filtered Pandas datafram with only astronauts within the
        group window.
    
    """
    below = nasa_astronaut_dataset[nasa_astronaut_dataset.Group <= group_max]
    above = below[below.Group >= group_min]
    return above

def frequency(nasa_astronaut_dataset,column):
    """
    Sorts occurences of cells in a column from most common to least common
    and how many time each element occurs.

    When used on columns with blank cells, blank cells are excluded. If an
    element has multiple elements, like a cell with two colleges or majors
    listed, it splits them and counts them each.

    Args:
        nasa_astronaut_dataset: A pandas dataframe containing information
        from the 2013 NASA Astronaut Factbook.

        column: An integar representing the column to count.

    Returns:
        new: A dictionary representing the most common elements ordered
        most frequent elements and least common elements and how many
        times they occur in the column.
        
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
    Returns top key-value pairs in a dictionary.

    Args:
        new: An dictionary from the frequency function
        representing an ordered list of elements from
        a column.

        top_number: An integar representing the number of items
        to extract.
    """
    return dict(itertools.islice(new.items(), top_number))

def engineer(nasa_astronaut_dataset):
    """
    Says how many many NASA astronauts had some kind of engineering
    undergraduate major.

    Args:
        nasa_astronaut_dataset: A pandas dataframe containing information
        from the 2013 NASA Astronaut Factbook.

    """
    new = frequency(nasa_astronaut_dataset, 8)

    majors = new.keys()

    engineer = 0
    non = 0
    for major in majors:
        if "Engineering" in major or "engineering" in major:
            engineer += 1
            continue
        non +=1
    total = engineer + non
    print(f"{engineer / total * 100} % of Astronauts majored"
          " in some kind of Engineering")

def average(nasa_astronaut_dataset, column):
    """
    Returns the average of a column in a daframe.

    Args:
        nasa_astronaut_dataset: A pandas dataframe containing information
        from the 2013 NASA Astronaut Factbook.

        column: An integar representing the column to take the average of.

    Returns:
        A float representing the mean of the column.
    """
    return nasa_astronaut_dataset[column].mean()


def plot_astronauts_vs_time(nasa_astronaut_dataset):
    """
    Plots the number of NASA astronauts going into space over the years.

    Does not include people without an official year listed, like payload
    specialists.

    Args:
        nasa_astronaut_dataset: A pandas dataframe containing information
        from the 2013 NASA Astronaut Factbook.

    """
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


def grad_school_vs_not_grad_school(nasa_astronaut_dataset):
    """
    Returns how many people in the nasa astronaut dataset went
    to graduate school and how many people did not.

    Args:
        nasa_astronaut_dataset: A pandas dataframe containing information
        from the 2013 NASA Astronaut Factbook.

    Returns:
        a list containing the number of people who went to and did not
        go to gradschool.
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
    """
    Plots minimum, average, and maximum age of astronauts in each astronaut
    group. 

    Does not take into account those without an official group number.

    Args:
        nasa_astronaut_dataset: A pandas dataframe containing information
        from the 2013 NASA Astronaut Factbook.

    """
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

def most_common_state(nasa_astronaut_dataset):
    """
    Plots an interactive map heat map showing how many astronauts
    were born from each state.

    Args:
        nasa_astronaut_dataset: A pandas dataframe containing information
        from the 2013 NASA Astronaut Factbook.

    """
    most_comon_states = {}
    frequent_state = frequency(nasa_astronaut_dataset, 19)
    for key in frequent_state:
        if key.isupper():
            most_comon_states[key] = frequent_state[key]
            
    most_comon_states_keys =list( most_comon_states.keys() )
    most_comon_numbers =list( most_comon_states.values() )

    state_df = pandas.DataFrame(list(zip(most_comon_states_keys, most_comon_numbers)),
               columns =['state', 'value'])
    
    import plotly.express as px  # Be sure to import express
    fig = px.choropleth(state_df,  # Input Pandas DataFrame
                        locations="state",  # DataFrame column with locations
                        color="value",  # DataFrame column with color values
                        hover_name="state", # DataFrame column hover info
                        locationmode = 'USA-states') # Set to plot as US States
    fig.update_layout(
        title_text = 'Most Popular Astronaut Home States', # Create a Title
        geo_scope='usa',  # Plot only the USA instead of globe
    )
    fig.show()  # Output the plot to the screen
    
def female_astronauts_decade(nasa_astronaut_dataset):
    """
    Plots NASA astronaut groups and what percentage of each were women.

    Args:
        nasa_astronaut_dataset: A pandas dataframe containing information
        from the 2013 NASA Astronaut Factbook.

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
    """
    Plots NASA astronaut groups and what percentage of each came
    from a military college like the US Naval Academy or The Citadel.

    Args:
        nasa_astronaut_dataset: A pandas dataframe containing information
        from the 2013 NASA Astronaut Factbook.

    """
    values = []
    military_experience = []
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
        values.append(np.nan)
    group = range(23)
    plt.scatter(group,values)
    plt.xlabel('Group Number')
    plt.ylabel('% From Military College')
    plt.title('Military Education in Austronauts over Time')
    
def top_college_over_time(nasa_astronaut_dataset):
    """
    Lists the college the most astronauts came from for each
    official astronaut group.

    Args:
        nasa_astronaut_dataset: A pandas dataframe containing information
        from the 2013 NASA Astronaut Factbook.

    Returns:
        values: a list containing the most common college from each astronaut
        group in order.
    """
    values = []
    for i in range(23):
        grouper = nasa_astronaut_dataset[nasa_astronaut_dataset.Group == i]
        frequency_of_colleges = frequency(grouper, 7)
        keys = list(frequency_of_colleges.keys())
        if len(keys) > 0:
            values.append(keys[0])
        
    return values

def grad_school_over_time(nasa_astronaut_dataset):
    """
    Plots pie charts showing what percentage of astronauts went to graduate
    school over time.

    Note that while astronauts that did not have an official group or year
    were not included, they most likely did considering they were probably
    selected from the research community.

    Args:
        nasa_astronaut_dataset: A pandas dataframe containing information
        from the 2013 NASA Astronaut Factbook.

    """
    
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

def gender_military(nasa_astronaut_dataset, gender, title):
    """
    Creates a bar graph showing how many men/women there were in the
    NASA astronaut dataset and how many of each were military
    vs. civilians.

    Args:
        nasa_astronaut_dataset: A pandas dataframe containing information
        from the 2013 NASA Astronaut Factbook.

    """
    gender_occurrence = nasa_astronaut_dataset.groupby('Gender').count()
    gender_military = gender_occurrence["Military Rank"]
    gender_occurrence_name = gender_occurrence["Name"]

    count_row = nasa_astronaut_dataset.shape[0]
    plt.style.use("ggplot")
    plt.bar(gender, gender_occurrence_name, width=0.8, label='Civilian', color='silver')
    plt.bar(gender, gender_military, width = 0.8, label = 'Military', color = 'gold')
    plt.xlabel("Gender")
    plt.ylabel("Number of Astronauts")
    plt.title(title)

    plt.legend(loc="upper left")
    plt.show()

def first_v_last(nasa_astronaut_dataset):
    """
    Plots a gender and military bar graph for first and last astronaut group.

    Args:
        nasa_astronaut_dataset: A pandas dataframe containing information
        from the 2013 NASA Astrounat Factbook.

    """
    first = filter_by_group(nasa_astronaut_dataset, 1, 1)

    last = filter_by_group(nasa_astronaut_dataset, 20, 20)
    
    gender_military(first, ["Male"], "Gender Distribution of Astronauts in First Astronaut Class")
    
    gender_military(last, ["Female", "Male"], "Gender Distribution of Astronauts in Last Astronaut Class")