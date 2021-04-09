"""
Import visualization tools.
"""

import itertools
import matplotlib.pyplot as plt
import pandas
import numpy as np
from pylab import plot, title, xlabel, ylabel, legend, array
import plotly.express as px  # Be sure to import express


def change_dates(nasa_astronaut_dataset):
    """
    Takes NASA astronaut dataframe adds each astronauts selection age to a
    column based on their birth date and selection year.

    Args:
        nasa_astronaut_dataset: A pandas dataframe containing information
        from the 2013 NASA Astronaut Factbook.

    Returns:
        nasa_astronaut_dataset: Pandas Dataframe with added "Selection Age"
        column
    """
    nasa_astronaut_dataset['Birth Date'] = (
        nasa_astronaut_dataset['Birth Date'].str[-4:])
    nasa_astronaut_dataset["Birth Date"] = (
        nasa_astronaut_dataset["Birth Date"].astype(float))
    nasa_astronaut_dataset["Selection Age"] = (
        nasa_astronaut_dataset["Year"]
        - nasa_astronaut_dataset["Birth Date"])

    return nasa_astronaut_dataset


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

        group_min: An integar representing the group cutoff.

        group_max: An integar representing the group cutoff.

    Returns:
        above: A filtered Pandas datafram with only astronauts within the
        group window.

    """
    below = nasa_astronaut_dataset[nasa_astronaut_dataset.Group <= group_max]
    above = below[below.Group >= group_min]
    return above


def frequency(nasa_astronaut_dataset, column):
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
    colleges = nasa_astronaut_dataset.iloc[:, column]
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
#    new = {k: v for k, v in sorted(college_frequency.items(),
#                                 key=lambda item: item[1], reverse=True)}
    new = dict(sorted(college_frequency.items(), key=lambda item: item[1],
                      reverse=True))
    new.pop("nan", None)
    return new


def horizontal_bar(nasa_astronaut_dataset, column, title_name):
    """
    Creates horizonat bar graph of top 25 most common cells in a column.

    Args:
        nasa_astronaut_dataset: A pandas dataframe containing information
        from the 2013 NASA Astronaut Factbook.

        column: An integar representing what column to count with the
        frequency function.

        title_name: A string representing what column the chart should
        be titled by.
    """
    plt.rcdefaults()
    _, axis = plt.subplots()
    new = frequency(nasa_astronaut_dataset, column)
    toppers = tops(new, 25)
    colleges = list(toppers.keys())
    numbers = list(toppers.values())

    axis.barh(colleges, numbers, align='center', height=.8)
    axis.invert_yaxis()  # labels read top-to-bottom
    axis.set_xlabel('Number of Astronauts per Major')
    axis.set_title(f'Top 25 Most Common Astronaut {title_name}')

    plt.show()


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

    engineer_count = 0
    non = 0
    for major in majors:
        if "Engineering" in major or "engineering" in major:
            engineer_count += 1
            continue
        non += 1
    total = engineer_count + non
    print(f"{engineer_count / total * 100} % of Astronauts majored"
          " in some kind of Engineering")


def plot_astronauts_vs_time(nasa_astronaut_dataset):
    """
    Plots the number of NASA astronauts going into space over the years.

    Does not include people without an official year listed, like payload
    specialists.

    Args:
        nasa_astronaut_dataset: A pandas dataframe containing information
        from the 2013 NASA Astronaut Factbook.

    """
    new = frequency(nasa_astronaut_dataset, 1)
    new = dict(sorted(new.items(), key=lambda item: item[0]))

    years = list(new.keys())
    astronauts = list(new.values())

    years = list(map(float, years))
    astronauts = list(map(float, astronauts))

    x_ticks = np.arange(min(years), max(years), 5)
    plt.xticks(x_ticks)
    plt.plot(years, astronauts)
    plt.xlabel("Year")
    plt.ylabel("Number of Astronauts")
    plt.title("Number of NASA Astronauts over the Years")


def grad_school_vs_not_grad_school(nasa_astronaut_dataset, column):
    """
    Returns how many people in the nasa astronaut dataset went
    to graduate school and how many people did not.

    Args:
        nasa_astronaut_dataset: A pandas dataframe containing information
        from the 2013 NASA Astronaut Factbook.

        column: An integar representing which column to count.
    Returns:
        a list containing the number of people who went to and did not
        go to gradschool.
    """
    gradschool = 0
    not_gradschool = 0
    grad_or_not = list(nasa_astronaut_dataset.iloc[:, column])
    for i in grad_or_not:
        if isinstance(i, str):
            gradschool += 1
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

    most_comon_states_keys = list(most_comon_states.keys())
    most_comon_numbers = list(most_comon_states.values())

    state_df = pandas.DataFrame(list(zip(most_comon_states_keys, most_comon_numbers)),
                                columns=['state', 'value'])

    fig = px.choropleth(state_df,  # Input Pandas DataFrame
                        locations="state",  # DataFrame column with locations
                        color="value",  # DataFrame column with color values
                        hover_name="state",  # DataFrame column hover info
                        locationmode='USA-states')  # Set to plot as US States
    fig.update_layout(
        title_text='Most Popular Astronaut Home States',  # Create a Title
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
    plt.step(decade, frequency_per_decade, color='blue')
    plt.xlabel('Decade')
    plt.ylabel('% Female Astronauts (pct.)')
    plt.title('Female Astronauts Selected Per Decade')

    plt.show()


def female_per_year(nasa_astronaut_dataset):
    """
    Find the total number of female astronauts per selection group.

    Args:
        nasa_astronaut_dataset: pandas dataset.

    Returns:
        female_per_year: a list of the number of females per selection year.
    """
    counter = 1
    female_per_year_list = []

    while counter <= 20:
        new_set = filter_by_group(nasa_astronaut_dataset, counter, counter)
        frequency_dict = frequency(new_set, 6)
        if 'Female' in frequency_dict:
            num_females = frequency_dict.get('Female')
            female_per_year_list.append(num_females)
        else:
            female_per_year_list.append(0)
        counter += 1

    return female_per_year_list


def astronauts_per_group(nasa_astronaut_dataset):
    """
    Creates a list of the number of astronauts per selection group.

    Args:
        nasa_astronauts_dataset: a pandas dataframe.

    Returns:
        astronauts_per_group: a list of the number of astronauts
        per selection group.
    """
    counter = 1
    astronauts_per_group_list = []

    while counter <= 20:
        new_set = filter_by_group(nasa_astronaut_dataset, counter, counter)
        frequency_dict = frequency(new_set, 6)
        values = frequency_dict.values()
        total = sum(values)
        astronauts_per_group_list.append(total)
        counter += 1

    return astronauts_per_group_list


def military_college_over_time(nasa_astronaut_dataset):
    """
    Plots NASA astronaut groups and what percentage of each came
    from a military college like the US Naval Academy or The Citadel.

    Args:
        nasa_astronaut_dataset: A pandas dataframe containing information
        from the 2013 NASA Astronaut Factbook.

    """
    values = []
    military_affiliation = []
    group = range(23)
    for i in group:
        grouper = nasa_astronaut_dataset[nasa_astronaut_dataset.Group == i]

        college_list = []
        colleges = grouper["Alma Mater"]
        for j in colleges:
            i = str(i)
            split_list = j.split("; ")
            college_list += split_list

        military_prep = 0
        non = 0
        for k in college_list:
            if "US " not in k and "The Citadel" not in k:
                military_prep += 1
            else:
                non += 1
        if (military_prep + non) > 0:
            values.append(military_prep / (military_prep + non) * 100)
        else:
            values.append(np.nan)
        affiliation = grad_school_vs_not_grad_school(grouper, 11)
        if affiliation[0] + affiliation[1] > 0:
            military_affiliation.append(affiliation[0] /
                                        (affiliation[0] + affiliation[1])*100)
        else:
            military_affiliation.append(np.nan)

    plt.plot(group, values, color='blue', label='Military Education')
    plt.fill_between(group, values, color='blue')
    plt.plot(group, military_affiliation, color='red',
             label='Military Affiliaiton')
    plt.fill_between(group, military_affiliation, color='red')
    plt.xticks(np.arange(min(group), max(group)+1, 1.0))
    plt.xlabel('Selection Groups')
    plt.ylabel('% of Astronauts')
    plt.title("Percentage Military Education and Military"
              " Affiliation Over Group")
    plt.legend(loc="upper left", prop={'size': 8.5})
    plt.show()

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
    inthesix = grad_school_vs_not_grad_school(sixties, 9)

    seventies = filter_by_year(nasa_astronaut_dataset, 1970, 1980)
    intheseven = grad_school_vs_not_grad_school(seventies, 9)

    eighties = filter_by_year(nasa_astronaut_dataset, 1980, 1990)
    intheeight = grad_school_vs_not_grad_school(eighties, 9)

    ninties = filter_by_year(nasa_astronaut_dataset, 1990, 2014)
    inthenine = grad_school_vs_not_grad_school(ninties, 9)

    labels = 'Gradschool', 'Not'

    fig, axs = plt.subplots(2, 2)
    axs[0, 0].pie(inthesix, labels=labels, autopct='%1.1f%%')
    axs[0, 0].set_title('1960s')
    axs[0, 1].pie(intheseven, labels=labels, autopct='%1.1f%%')
    axs[0, 1].set_title('1970s')
    axs[1, 0].pie(intheeight, labels=labels, autopct='%1.1f%%')
    axs[1, 0].set_title('1980s')
    axs[1, 1].pie(inthenine, labels=labels, autopct='%1.1f%%')
    axs[1, 1].set_title('90s and 2000s')

    fig.suptitle('Astronaut Post-Grad Trends through the Decades',
                 x=.5, y=1, fontsize=13)

    # Hide x labels and tick labels for top plots and y ticks for right plots.
    for axis in axs.flat:
        axis.label_outer()


def gender_military(nasa_astronaut_dataset, gender, title_name):
    """
    Creates a bar graph showing how many men/women there were in the
    NASA astronaut dataset and how many of each were military
    vs. civilians.

    Args:
        nasa_astronaut_dataset: A pandas dataframe containing information
        from the 2013 NASA Astronaut Factbook.

        gender: A list of strings representing the genders to count for and
        in what order.

        title_name: A string representing the title of the plot.

    """
    gender_occurrence = nasa_astronaut_dataset.groupby('Gender').count()
    gender_in_military = gender_occurrence["Military Rank"]
    gender_occurrence_name = gender_occurrence["Name"]

    plt.style.use("ggplot")
    plt.bar(gender, gender_occurrence_name,
            width=0.8, label='Civilian', color='silver')
    plt.bar(gender, gender_in_military, width=0.8,
            label='Military', color='gold')
    plt.xlabel("Gender")
    plt.ylabel("Number of Astronauts")
    plt.title(title_name)

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

    gender_military(first, ["Male"],
                    "Gender Distribution of Astronauts in"
                    "First Astronaut Class")

    gender_military(last, ["Female", "Male"],
                    "Gender Distribution of Astronauts in"
                    "Last Astronaut Class")


def female_and_total(nasa_astronaut_dataset):
    """
    Creates a line plot with the total amount
    of astronauts selected per group and the total number
    of females per group.

    Args:
        nasa_astronaut_dataset: a pandas dataset.

    Returns:
        A plot of astronauts selected per group alongside
        female astronauts selected per group.
    """
    females = female_per_year(nasa_astronaut_dataset)
    total_per_year = astronauts_per_group(nasa_astronaut_dataset)
    group = []
    for i in range(20):
        group.append(i+1)
    plt.plot(group, total_per_year, color='purple', label='Total Astronauts')
    plt.fill_between(group, total_per_year, color='purple')
    plt.plot(group, females, color='pink', label='Female Astronauts')
    plt.fill_between(group, females, color='pink')
    plt.xticks(np.arange(min(group), max(group)+1, 1.0))
    plt.xlabel('Selection Groups')
    plt.ylabel('Number of Astronauts')
    plt.title("Number of astronauts per Selection"
              " Group and Number of Females per Group")
    plt.legend(loc="upper left", prop={'size': 8.5})
    plt.show()
