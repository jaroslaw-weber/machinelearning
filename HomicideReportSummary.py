import seaborn as sns
from sklearn import preprocessing, ensemble
from scipy.stats import kendalltau
import pandas as pd
import random

#### Analysing US homicide reports with simple summaries and graphs.
# Data is available at:
# https://www.kaggle.com/murderaccountability/homicide-reports


#simple formatting
ll="____________________________"
def pll():
    print(ll)

#import csv file with data
csv=pd.read_csv("homicide-reports.csv")

#print data info
pll()
print("--> data loaded. length: {0}".format(csv.size))
print("columns:")
print(csv.columns)
pll()

##select useful columns
#select categorical data
cats=csv[['Victim Sex','Victim Race', 'Victim Ethnicity','Perpetrator Sex',
          'Perpetrator Race', 'Perpetrator Ethnicity' , 'Relationship']]
#select numeric data
nums=csv[['Perpetrator Age', 'Victim Age']]

#numeric data parse function
def to_num(data_to_parse):
    numeric = pd.to_numeric(data_to_parse, errors='coerce')
    return numeric.dropna().where(lambda a: a > 10).where(lambda a: a < 70)

#parse victim and perpetrator age columns
pa=to_num(nums['Perpetrator Age'])
va=to_num(nums['Victim Age'])

#use visualisations to show victim/perpetrator age correlation
with sns.axes_style("white"):
    hex_plt=sns.jointplot(x=va, y=pa, kind="hex", gridsize=24, space=0, color="r")
    print("--> plotting data on hex graph....")
    sns.plt.show()

#describe categorical data (frequency/count)
pll()
print("--> categorical data summaries:")

def dsc(label, series):
    pll()
    print("{0}:".format(label))
    d=series.describe()
    s=d.sort(['counts'], ascending=False)
    print(s)

for i in range(len(cats.columns)):
    column_name=cats.columns[i]
    column_data=cats[column_name]
    parsed=pd.Categorical(column_data)
    dsc(column_name,parsed)
