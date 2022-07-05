import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from imblearn.over_sampling import SMOTE
from sklearn import preprocessing
from streamlit_option_menu import option_menu
#import tkinter
st.set_option('deprecation.showPyplotGlobalUse', False)



Menu = option_menu(None, ["Dataset","Dashboard", "Results"],icons=["cloud","bar-chart-line","clipboard-check"],menu_icon="cast", default_index=0, orientation="horizontal", styles={"container": {"padding": "0!important", "background-color": "#fafafa"},"icon": {"color": "black", "font-size": "25px"}, "nav-link": {"font-size": "15px", "text-align": "left", "margin":"0px", "--hover-color": "#eee"},"nav-link-selected": {"background-color": "pink"},})
if Menu == "Dataset": st.title('Heart Stroke Dataset')
if Menu == "Dashboard": st.title('Heart Stroke Dashboard')
if Menu == "Recommendations" : st.title("Results")


df= pd.read_csv("healthcare-dataset-stroke-data.csv")
if Menu=="Dataset": st.write(df)

#header = st.container()

#with header: 
    #st.title("Analyzing Heart Strokes")

# Examine the dataset
#df.shape
df.info()
df.describe()


df.drop(df[df.gender == 'Other'].index, inplace = True)

# Exploring missing data
#plt.style.use("seaborn")
#plt.figure(figsize=(10,5))
#fig1 = sns.heatmap(df.isnull(), yticklabels = False, cmap = 'plasma')
#plt.title('Null Values in Data Frame')
#st.pyplot(fig1)


# get the number of missing data points per column

missing_value_count = (df.isnull().sum())
print(missing_value_count[missing_value_count > 0])

# percent of data that is missing

total_cells = np.product(df.shape)
total_missing_value = missing_value_count.sum()
print('Percentage of missing value in Data Frame is:', total_missing_value / total_cells*100)
print('Total number of our cells is:', total_cells)
print('Total number of our missing value is:', total_missing_value)


# Handling missing values

df['bmi'].fillna(df['bmi'].mean(),inplace=True)
df['bmi'].isnull().sum()



object_col = ["gender", "ever_married" ,"Residence_type","work_type","smoking_status"]
label_encoder = preprocessing.LabelEncoder()
for col in object_col:
    df[col]=  label_encoder.fit_transform(df[col])


X = df.drop(columns = ['stroke'])
y = df['stroke']

# SMOTE technique to balance out the dataset
sm = SMOTE(random_state=123)
X_sm , y_sm = sm.fit_resample(X,y)

print(f'''Shape of X before SMOTE:{X.shape}
Shape of X after SMOTE:{X_sm.shape}''',"\n\n")

print(f'''Target Class distributuion before SMOTE:\n{y.value_counts(normalize=True)}
Target Class distributuion after SMOTE :\n{y_sm.value_counts(normalize=True)}''')



# Labeling data fields to Text value for easy interpretation of Visualization
data_eda = df.copy()
#hypertension: 0 if the patient doesn't have hypertension, 1 if the patient has hypertension
data_eda["hypertension"]     = df["hypertension"]    .map({1: "Yes",           0: "No"})
#1 if the patient had a stroke or 0 if not
data_eda["stroke"]     = df["stroke"]    .map({1: "Yes",           0: "No"})
#0 if the patient doesn't have any heart diseases, 1 if the patient has a heart disease
data_eda["heart_disease"]  = df["heart_disease"] .map({1: "Yes" ,           0: "No"})




def age_cohort(age):
    if   age >= 0 and age <= 20:
        return "0-20"
    elif age > 20 and age <= 40:
        return "20-40"
    elif age > 40 and age <= 50:
        return "40-50"
    elif age > 50 and age <= 60:
        return "50-60"
    elif age > 60:
        return "60+"
    
data_eda['age group'] = data_eda['age'].apply(age_cohort)
data_eda.sort_values('age group', inplace = True)




def pie_graph(df,title,values):   
    labels = df[values].value_counts().index
    values = df[values].value_counts()

    fig = go.Figure(data = [
        go.Pie(
        labels = labels,
        values = values,
        hole = .5)
    ])

    fig.update_layout(title_text = title)
    if Menu=="Dashboard":st.write(fig)



#distribution_list = ["age","gender","hypertension","heart_disease","ever_married","work_type","stroke"]
#distributionl = st.selectbox('Choose an option:', options= distribution_list)
#dfage = data_eda[data_eda["age"] == distributionl]
#dfgender = data_eda[data_eda["gender"] == distributionl]



if Menu=="Dashboard":st.header("What are the distribution of features in data? How are they related to Heart Stroke?")



age = pie_graph(data_eda,"Age Group Distribution",'age group')
if Menu=="Dashboard": st.write(age)

gender =pie_graph(data_eda, 'Gender Distribution','gender')
if Menu=="Dashboard": st.write(gender)


hypertension = pie_graph(data_eda, 'Hypertension Distribution','hypertension')
if Menu=="Dashboard":st.write(hypertension)


heart = pie_graph(data_eda, 'Heart disease Distribution','heart_disease')
if Menu=="Dashboard":st.write(heart)


married = pie_graph(data_eda, 'Ever married  Distribution','ever_married')
if Menu=="Dashboard":st.write(married)


stroke = pie_graph(data_eda, 'Stroke Distribution', 'stroke')
if Menu=="Dashboard":st.write(stroke)


work = pie_graph(data_eda, 'Work type Distribution','work_type')
if Menu=="Dashboard":st.write(work)

def count_bar_plot(df,x,hue,title):
    fig = sns.countplot(x=x, hue=hue, data=df)
    fig.set_title(title)



gender_stroke = count_bar_plot(data_eda,'gender','stroke','Stroke distribution by Gender')
if Menu=="Dashboard":st.pyplot(gender_stroke)


hypertension_stroke = count_bar_plot(data_eda,'hypertension','stroke','Stroke distribution by hypertension')
if Menu=="Dashboard":st.pyplot(hypertension_stroke)

heart_stroke = count_bar_plot(data_eda,'heart_disease','stroke','Stroke distribution by heart_disease')
if Menu=="Dashboard":st.pyplot(heart_stroke)


married_stroke = count_bar_plot(data_eda,'ever_married','stroke','Stroke distribution by ever married')
if Menu=="Dashboard":st.pyplot(married_stroke)


residence_stroke = count_bar_plot(data_eda,'Residence_type','stroke','Stroke distribution by Residence type')
if Menu=="Dashboard":st.pyplot(residence_stroke)


def horizontal_bar_chart(df,x,y,color,title):    
    fig = px.bar(df, x=x, y=y, color=color,                  
                 height=600,
                 title=title)
    if Menu=="Dashboard":st.write(fig)
    

group = data_eda.groupby(['stroke','work_type'],as_index = False).size().sort_values(by='size')
work_type = horizontal_bar_chart(df = group,x = 'stroke',y = 'size',color = 'work_type',title = 'Stroke distribution by work type')
if Menu=="Dashboard":st.write(work_type)



group = data_eda.groupby(['stroke','smoking_status'],as_index = False).size().sort_values(by='size')
smoke_stroke = horizontal_bar_chart(df = group,x = 'stroke',y = 'size',color = 'smoking_status',title = 'Stroke distribution by smoking status')
if Menu=="Dashboard":st.write(smoke_stroke)


f, ax = plt.subplots(figsize = (12,10))
sns.heatmap(df.corr(),
            annot = True,
            linewidths = .5,
            fmt = '.1f',
            ax = ax)
if Menu=="Dashboard":st.pyplot(f)



# Average glucose level as per age 
if Menu=="Dashboard":f, age_glucose = plt.subplots(1,1, figsize=(10,8))
if Menu=="Dashboard": age_glucose = sns.scatterplot(data = df , x = 'age' , y ='avg_glucose_level',hue='bmi').set(title='Average Glucose Level & BMI as per Age')
if Menu=="Dashboard": st.pyplot(f)


# BMI as per age
if Menu=="Dashboard":f1, age_bmi = plt.subplots(1,1, figsize=(10,8))
if Menu=="Dashboard": age_bmi = sns.scatterplot(data = df , x = 'age' , y ='bmi')
if Menu=="Dashboard": st.pyplot(f1)
