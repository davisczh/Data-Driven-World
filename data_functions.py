import pandas as pd
from math import log10, floor
import numpy as np
import country_list as cl
import matplotlib.pyplot as plt


def read_file(file_path):       #argument takes in file path e.g. C:\SUTD Courses\Year 1 Term3\Modelling Uncertainty\pred1
    df = pd.read_csv(fr'{file_path}.csv')
    return df

#filter data based on specifications
def filter_based_on_spec(df, indicator, dimension, category, sex, age, year, unit, predictor):      #input specification
    # all arguments are string aside from df(csv file) and year(int array)

    result = df.loc[(df["Indicator"] == indicator) & (df["Dimension"] == dimension) & (df["Category"] == category)
                    & (df["Sex"] == sex) & (df["Age"] == age) & (df["Year"] >= year[0]) & (df["Year"] < year[-1])
                    & (df["Unit of measurement"] == unit), ["Country", "Year", "VALUE"]]
    output = {"Country": [], "Year": [], "VALUE": []}
    #filter by country
    for country in cl.country_list:
        for index in result.index[result["Country"] == country].tolist():
            output["Country"].append(result.at[index, "Country"])
            output["Year"].append(result.at[index, "Year"])
            output["VALUE"].append(result.at[index, "VALUE"])
    result = pd.DataFrame().from_dict(output)
    result[predictor] = result["VALUE"]

    return result   # return a reference, NOT a copy

def threeyearaverage(df, year):
    countries = []
    for country in df["Country"]:
        if country not in countries:
            countries.append(country)

    def extract_value(df, col):
        try:
            output = df.at[df.index[df['countryear1'] == col].tolist()[0], "VALUE"]
        except:
            output = None
        return output

    start_year = year[0]
    end_year = year[1]
    output = {"countryear": [], f"{df.columns[3]}": []}
    for country in countries:
        for year in range(start_year, end_year -1):
            countryear1 = country + str(year)
            countryear2 = country + str(year + 1)
            countryear3 = country + str(year + 2)
            year1v = extract_value(df, countryear1)
            year2v = extract_value(df, countryear2)
            year3v = extract_value(df, countryear3)
            if year1v and year2v and year3v:
                avg = find_average(year1v, year2v, year3v)
                countryear = f"{country}{year}-{year+2}"
                output["countryear"].append(countryear)
                output[f"{df.columns[3]}"].append(avg)
    return pd.DataFrame().from_dict(output)

def add_countryear(df):
    output = df.copy()
    output["countryear1"] = output["Country"] + output["Year"].astype(str)
    return output

def get_predictor(df, indicator, dimension, category, sex, age, year, predictor):
    counts = filter_based_on_spec(df, indicator, dimension, category, sex, age, year, "Counts", predictor)
    counts = add_countryear(counts)
    counts = threeyearaverage(counts, year)
    per100000 = filter_based_on_spec(df, indicator, dimension, category, sex, age, year, "Rate per 100,000 population", "per 100,000")
    per100000 = add_countryear(per100000)
    per100000 = threeyearaverage(per100000, year)


    output = {"countryear": [], predictor: [], "per 100,000": []}
    for countryear in counts["countryear"]:
        if countryear in list(per100000["countryear"]):
            output["countryear"].append(countryear)
            output[predictor].append(counts.at[counts.index[counts['countryear'] == countryear].tolist()[0], predictor])
            output["per 100,000"].append(per100000.at[per100000.index[per100000['countryear'] == countryear].tolist()[0], "per 100,000"])

    output = pd.DataFrame().from_dict(output)
    output = clean_dataset(output, predictor)

    return output


#recursion method used in filtering the years
def filter_years(df, years):
    if len(years) == 1:             # breakpoint for recursion
        return df["Year"] == years[-1]
    year_range = (df["Year"] == years[-1]) | filter_years(df, years[:-1])        # recursion for all elements in years, which is a string array

    return year_range       # e.g. year_range = (df["Year"]=="2019") | (df["Year"]=="2016")

#filter data using years and element
def get_target(df, element="Value", flag = "O", years=["all"]):          #years argument is a string array

    if years[0] == "all":
        result = df.loc[(df["Element"] == element), ["Area", "Year", "Value"]]
        return result

    year_range = filter_years(df, years)
    result = df.loc[(df["Flag"] != flag) & (df["Element"] == element) & year_range, ["Area", "Year", "Value"]]
    output = {"Area": [], "Year": [], "Value": []}
    for country in cl.country_list:
        for index in result.index[result["Area"] == country].tolist():
            output["Area"].append(result.at[index, "Area"])
            output["Year"].append(result.at[index, "Year"])
            output["Value"].append(result.at[index, "Value"])
    result = pd.DataFrame().from_dict(output)

    result["countryear"] = result["Area"] + result["Year"]
    result['Value'] = result['Value'].apply(lambda x:x if (x != "<0.1") else "0.1")
    result[['Value']] = result[['Value']].apply(pd.to_numeric)
    return result


def clean_dataset(df, predictor):

    q1 = df.iloc[:, 2].quantile(q=0.25)
    q3 = df.iloc[:, 2].quantile(q=0.75)
    iqr = q3 - q1
    min = q1 - (1.5*iqr)
    max = q3 + (1.5*iqr)


    result = df.loc[(df["per 100,000"] > min) & (df["per 100,000"] < max), ["countryear", predictor]]
    return result



def find_average(val1, val2, val3):
    result = (val1 + val2 + val3)/3

    def round_sig(value, sig=3):
        return round(value, sig - int(floor(log10(abs(value)))) - 1)  # rounding to __ significant figure

    result = round_sig(result, 6)      #rounding to 6 sf
    return result





def combine_country(faostat, pred):     #df1 is FAOSTAT and df2 is pred dataset
    area = faostat["Area"]
    area_df = pd.DataFrame(area)
    country = pred["Country"]
    filtered_countries = np.array([])
    for i in range(len(country)):
        print(area[i])
        # if area[i] == i:
        #     filtered_countries += area[i]

    # result_countries = pd.Series(filtered_countries)
    # return result_countries