"""
Perform Unit Tests on Functions
"""
# Import all required libraries.
import pytest
import pandas

# Import functions
from functions import (
    change_dates,
    frequency,
    tops,
    grad_school_vs_not_grad_school,
    filter_by_year,
    filter_by_group
)
# Import the test dataset
test_data = pandas.read_csv("Test_Data.csv")


@pytest.mark.parametrize("testing_data, passes_check", [
    # Test to Make Sure a column named "Selection Age" was made
    (test_data, True)
])
def test_changed_dates(testing_data, passes_check):
    """
    Check that the changed_dates function adds a selection_age column.

    Args:
        testing_data: A pandas dataframe representing the test data.

        passes_check: A boolean represeting what the assertion should be.
    """
    changed_dates = change_dates(testing_data)
    assert "Selection Age" in changed_dates


@pytest.mark.parametrize("dataset, column ,passes_check", [

    # Test that Year column returns Years with float values correctly.
    (test_data, 1, {'220.0': 2, '2004.0': 1}),

    # Test that Status column returns Years with string values correctly.
    (test_data, 3, {'Active': 2, 'Retired': 1}),

    # Test that an empty column returns an empty dictionary
    (test_data, 10, {}),

    # Test that an Alma Mater with multiple values gets split correctly
    (test_data, 7, {"Olin College": 2, "University of Arizona": 1,
                    "Wellesley": 1, "University of Colorado": 1,
                    "University of California-Santa Barbara": 1,
                    "Montana State University": 1}), ])
def test_frequency(dataset, column, passes_check):
    """
    Check that the frequency function returns the correct dictionaries.

    Args:
        dataset: A pandas dataframe representing the test data

        column: An integar representing the column to test

        passes_check: A bool representing the expected output of the checker.
    """
    assert frequency(dataset, column) == passes_check


# Making a solution dictionary
new = {"Olin College": 2, "University of Arizona": 1,
       "Wellesley": 1, "University of Colorado": 1,
       "University of California-Santa Barbara": 1,
       "Montana State University": 1}


@pytest.mark.parametrize("tester_data, top_number,passes_check", [

    # Make sure tops returns the key-value pair of the highest value
    # in a dictionary
    (new, 1, {"Olin College": 2}),

    # Makes sure tops returns the full dicitonary if top_number
    # is large than the number of key-value pairs
    (new, 10, new),

])
def test_tops(tester_data, top_number, passes_check):
    """
    Check to make sure the tops function behaves correctly

    Args:
        tester_data: A pandas dataframe representing the test data.

        top_number: An integar representing the number of items
        to pull from the dictionary.

        passes_check: A bool representing the expected output of the checker.
    """
    assert tops(tester_data, top_number) == passes_check


@pytest.mark.parametrize("test_dataset, column, passes_check", [

    # Make sure grad_school returns the right list for the test data.
    (test_data,9, [3, 1]),
])
def test_grad_school_vs_not_grad_school(test_dataset, column, passes_check):
    """
    Check to make sure grad_school behaves correctly.

    Args:
        test_dataset: A pandas dataframe representing the test data.

        passes_check: A bool representing the expected output of the checker.
    """
    assert grad_school_vs_not_grad_school(test_dataset,column) == passes_check

# Making solution dataframes


test_2 = test_data.loc[[2, 3], :]
blank_dataset = pandas.DataFrame(index=[], columns=test_data.columns)


@pytest.mark.parametrize("dataset, year_min, year_max, passes_check, checker", [
    # Make sure that setting the start and end date to one year chooses
    # only the entry for that year, test <= and >=.
    (test_data, 2004, 2004, pandas.DataFrame(test_data.loc[0, :]).T, True),

    # Make sure that the filter returns multiple rows when appropriate.
    (test_data, 220, 220, test_2, True),

    # Make sure filter doesn't include NaN years or anything outside filter years.
    (test_data, 2019, 2020, blank_dataset, True)
])
def test_filter_by_year(dataset, year_min, year_max, passes_check, checker):
    """
    Check to make sure filter by year behaves correctly.

    Args:
        dataset: A pandas dataframe representing the test data.

        year_min: An integar representing the year cutoff from below.

        year_max: An integar representing the year cutoff from above.

        passes_check: A pandas dataframe representing a solution data

        checker: A bool representign the expected output of the checker.
    """
    filtered_dataset = filter_by_year(dataset, year_min, year_max)
    checker = True
    assert filtered_dataset.shape[0] == passes_check.shape[0]

@pytest.mark.parametrize("dataset, group_min, group_max, passes_check, checker", [
    # Make sure that setting the start and end group to one year chooses
    # only the entry for that group, test <= and >=.
    (test_data, 19, 19, pandas.DataFrame(test_data.loc[0, :]).T, True),

    # Make sure that the filter returns multiple rows when appropriate.
    (test_data, 2024, 20204, test_2, True),

    # Make sure filter doesn't include NaN groups or anything
    #outside filter years.
    (test_data, 2019, 2020, blank_dataset, True)
])
def test_filter_by_group(dataset, group_min, group_max, passes_check, checker):
    """
    Check to make sure filter by group behaves correctly.

    Args:
        dataset: A pandas dataframe representing the test data.

        year_min: An integar representing the group cutoff from below.

        year_max: An integar representing the group cutoff from above.

        passes_check: A pandas dataframe representing a solution data

        checker: A bool representign the expected output of the checker.
    """
    filtered_dataset = filter_by_group(dataset, group_min, group_max)
    checker = True
    assert filtered_dataset.shape[0] == passes_check.shape[0]
