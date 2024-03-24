import os
import pandas as pd
import Levenshtein
import re


curerntPath = os.path.dirname(os.path.abspath(__file__))
data_file = os.path.join(curerntPath, "DataScientist.csv")


def get_list_of_items(df, column_name):
    items = []
    for index, row in df.iterrows():
        if (len(row[column_name]) > 0):
            for item in list(row[column_name]):
                if (type(item) is tuple and len(item) > 1):
                    item = item[0]
                if (item not in items):
                    items.append(item)
    return items


def get_emails(df):
    email_regex='[^\s:|()\']+@[a-zA-Z0-9\.]+\.[a-zA-Z]+'
    df['emails'] = df['Job Description'].apply(
    lambda x: re.findall(email_regex, x))
    emails = get_list_of_items(df, 'emails')
    return emails


def find_levenshtein(input_string, df):
    df['distance_to_' + input_string] = df['emails'].apply(lambda x:Levenshtein.distance(input_string, x))
    return df


def get_closest_email_lev(df, email):
    df = find_levenshtein(email, df)
    column_name = 'distance_to_' + email
    minimum_value_email_index = df[column_name].idxmin()
    email = df.loc[minimum_value_email_index]['emails']
    return email


df = pd.read_csv(data_file, encoding='utf-8')
emails = get_emails(df)
new_df = pd.DataFrame(emails,columns=['emails'])


input_string = "rohitt.macdonald@prelim.com"
email = get_closest_email_lev(new_df, input_string)
print(email)


def find_jaro(input_string, df):
    df['distance_to_' + input_string] = df['emails'].apply(lambda x: Levenshtein.jaro(input_string, x))
    return df


def get_closest_email_jaro(df, email):
    df = find_jaro(email, df)
    column_name = 'distance_to_' + email
    maximum_value_email_index = df[column_name].idxmax()
    email = df.loc[maximum_value_email_index]['emails']
    return email


df = pd.read_csv(data_file, encoding='utf-8')
emails = get_emails(df)
new_df = pd.DataFrame(emails,columns=['emails'])


input_string2 = "rohitt.macdonald@prelim.com"
email = get_closest_email_jaro(new_df, input_string2)
print(email)


print(Levenshtein.jaro_winkler("rohit.mcdonald@prolim.com", "rohit.mcdonald@prolim.org"))