"""Based on https://machinelearningmastery.com/machine-learning-in-python-step-by-step/"""
from builtins import print

import numpy
import pandas as pd


def print_overview(data_frame):
    """ Print pandas data frame overview"""
    print('## Data frame info:')
    print(data_frame.info())
    print('\n')

    print('## Data frame shape:')
    print(str(data_frame.shape[0]) + ' rows')
    print(str(data_frame.shape[1]) + ' columns')
    print('\n')

    print('## Data frame columns:')
    for column in data_frame.columns:
        print(column)
    print('\n')

    print('## Data head and tail:')
    print(data_frame.head(10))
    print('...')
    print(data_frame.tail(5))
    print('\n')

    print('Numeric values statistics:')
    # Note that float format is set
    pd.set_option('float_format', '{:f}'.format)
    print(data_frame.describe())
    print('\n')

def print_state(df):
    print(df.groupby('State').size(), '\n')

def print_state1(data_frame):
    map = {"XX" : 0}
    map.clear()

    for st in data_frame.get('State'):
        if map.__contains__(st):
            map[st]=map[st] + 1
        else:
            map[st] = 0
    map.pop("State")
    for st in map:
        print(st, map[st])

def print_Type_of_breach(df):
    print('Type of Breach:', df['Type of Breach'].unique(), '\n')

def print_Type_of_breach1(data_frame):
    set = {""}
    set.clear()
    for st in data_frame.get('Type of Breach'):
        commasplit = str(st).split(",")
        for aftercommasplit in commasplit:
            co = str(aftercommasplit).split("/")
            for c in co:
                if c[0] == ' ':
                    set.add(c[1:])
                else:
                    set.add(c)
    for st in set:
        print(st)
def txs_hack_it_data_frame(data_frame):
    pd.options.display.max_rows = None
    texas_hacking = df[(df['State'] == 'TX') & (df['Type of Breach'].str.contains('Hacking/IT Incident'))]
    return texas_hacking


def txs_hack_it_data_frame_w_print(data_frame):
    #new_data_frame = data_frame[data_frame['State'] == 'TX']
    #mew_new_data_frame = new_data_frame[new_data_frame['Type of Breach'] == 'Hacking/IT Incident']
    #print(new_data_frame)
    #print(mew_new_data_frame)
    #return mew_new_data_frame
    # Remove maximum printed rows limit. Otherwise next print can be truncated
    pd.options.display.max_rows = None
    texas_hacking = df[(df['State'] == 'TX') & (df['Type of Breach'].str.contains('Hacking/IT Incident'))]
    print(texas_hacking)
    return texas_hacking
def extract_Individual_Affected_w_print(data_frame):
    new_data_frame = txs_hack_it_data_frame(data_frame)
    new_data_frame = new_data_frame['Individuals Affected']
    print(new_data_frame)
    return new_data_frame

def extract_Individual_Affected(data_frame):
    new_data_frame = txs_hack_it_data_frame(data_frame)
    new_data_frame = new_data_frame['Individuals Affected']
    return new_data_frame

def mean_affected_individals(df):
    individuals_affected = extract_Individual_Affected(df)
    individuals_affected_mean = individuals_affected.mean()
    print('Texas hacking affected individuals per breach mean:',
          individuals_affected_mean)

def median_affected_individals(df):
    individuals_affected = extract_Individual_Affected(df)
    individuals_affected_median = individuals_affected.median()
    print('Texas hacking affected individuals per breach median:',
          individuals_affected_median)

def mode_affected_individals(df):
    individuals_affected = extract_Individual_Affected(df)
    individuals_affected_mode = individuals_affected.mode()
    print('Texas hacking affected individuals per breach mode:',
          individuals_affected_mode)

def standard_deviation_affected_individals(df):
    individuals_affected = extract_Individual_Affected(df)
    ind_aff = []
    for individual in individuals_affected:
        ind_aff = float(individual)
    individuals_affected_std = numpy.std(ind_aff)
    print('Texas hacking affected individuals per breach std derivation:',
          individuals_affected_std)


def quant_tex_hack(df):
    texas_hacking = txs_hack_it_data_frame(df)
    texas_hacking['Individuals Affected'] = pd.to_numeric(texas_hacking['Individuals Affected'])

    individuals_affected_quartiles = texas_hacking['Individuals Affected'].quantile([
        0.25, 0.5, 0.75])
    print('Texas hacking affected individuals per breach quartiles:')
    print(individuals_affected_quartiles.to_string())
    print('\n')


if __name__ == '__main__':
    names = ['Name of Covered Entity','State','Covered Entity Type','Individuals Affected','Breach Submission Date','Type of Breach','Location of Breached Information','Business Associate Present','Web Description']
    df = pd.read_csv('C:\\Users\\achil\Google Drive\\TUM\\Semester 7 Erasmus\\Machine Learning\\Übungen\\ics0030-machine-learning\\lab1-template-main\\data\\breach_report.csv', names = names)
    #print_overview(df)
    #print_state(df)
    #print_state1(df)
    #print_Type_of_breach(df)
    #txs_hack_it_data_frame(df)
    #extract_Individual_Affected(df)
    #mean_affected_individals(df)
    #median_affected_individals(df)
    #mode_affected_individals(df)
    #standard_deviation_affected_individals(df)
    quant_tex_hack(df)
