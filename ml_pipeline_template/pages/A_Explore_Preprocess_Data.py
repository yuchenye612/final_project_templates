import streamlit as st
import pandas as pd
import plotly.express as px
from pandas.plotting import scatter_matrix
import os
from itertools import combinations
import numpy as np
from sklearn.preprocessing import OrdinalEncoder
from helper_functions import fetch_dataset

#README
#Delete no location entry
#Delete unspecified accident cause
#Delete accidents uninvolved with bikes and motorcycle
#Clean unrelated features
#Visualization with sidebars
#Categorizing data for machine learning:
    #Instead of displaying accident leading cause as strings, create
    #more columns to represent the cause of accidient using boolean,
    #resulting in a sparse boolean matrix. Run code to view.
#A map

def sidebar_filter(df, chart_type, x=None, y=None):
    """
    This function renders the feature selection sidebar 

    Input: 
        - df: pandas dataframe containing dataset
        - chart_type: the type of selected chart
        - x: features
        - y: targets
    Output: 
        - list of sidebar filters on features
    """
    side_bar_data = []
    select_columns = []
    if (x is not None):
        select_columns.append(x)
    if (y is not None):
        select_columns.append(y)
    if (x is None and y is None):
        select_columns = list(df.select_dtypes(include='number').columns)

    for idx, feature in enumerate(select_columns):
        try:
            f = st.sidebar.slider(
                str(feature),
                float(df[str(feature)].min()),
                float(df[str(feature)].max()),
                (float(df[str(feature)].min()), float(df[str(feature)].max())),
                key=chart_type+str(idx)
            )
        except Exception as e:
            print(e)
        side_bar_data.append(f)

    return side_bar_data
#############################################

st.markdown("# Practical Applications of Machine Learning (PAML)")

#############################################

st.markdown("### Final Project - Pope -Ye")

#############################################

st.markdown('# Explore & Preprocess Dataset')

#############################################
df = None
df = fetch_dataset()

if df is not None:
    # Display original dataframe
    st.markdown('View initial data with missing values or invalid inputs')
    st.markdown('You have uploaded the dataset.')

    st.dataframe(df)

    
    # Deal with missing values
    st.markdown('### Handle missing locations')

    def remove_nans(df):
        """
        This function removes all NaN values in the dataframe

        Input: 
            - df: pandas dataframe
        Output: 
            - df: updated df with no Nan observations
        """
        # Remove obs with nan values
        df = df.dropna(subset=['LATITUDE'])
        # df.to_csv('remove_nans.csv',index= False)
        st.session_state['df'] = df
        return df
    df = remove_nans(df)
    st.write(df)
    
    st.markdown('### Handle unspecified accident cause')

    def remove_nans(df):
        """
        This function removes all NaN values in the dataframe

        Input: 
            - df: pandas dataframe
        Output: 
            - df: updated df with no Nan observations
        """
        # Remove obs with nan values
        df = df[df['CONTRIBUTING FACTOR VEHICLE 1']!='Unspecified']
        # df.to_csv('remove_nans.csv',index= False)
        st.session_state['df'] = df
        return df
    df = remove_nans(df)
    df = remove_nans(df)
    st.write(df)
    
    st.markdown('### Only look at bike and motorbike involved accidents')
    values = ['Motorcycle','Bike']
    df = df[(df['VEHICLE TYPE CODE 1'].isin(values)) | (df['VEHICLE TYPE CODE 2'].isin(values))| (df['VEHICLE TYPE CODE 3'].isin(values))]
    
    st.write(df)
    # Some feature selections/engineerings here
    st.markdown('### Remove Irrelevant/Useless Features')
    dfl = df
    cols = ['CRASH DATE','ZIP CODE','ON STREET NAME','CROSS STREET NAME','OFF STREET NAME',
            'COLLISION_ID','VEHICLE TYPE CODE 4','VEHICLE TYPE CODE 5',
            'NUMBER OF PEDESTRIANS INJURED','NUMBER OF PEDESTRIANS KILLED',
            'BOROUGH', 'CRASH TIME', 
            'CONTRIBUTING FACTOR VEHICLE 4','CONTRIBUTING FACTOR VEHICLE 5',
            ]
    df = df.drop(labels=cols,axis=1)
    st.write(df)
    # Remove outliers
    st.markdown('### Remove outliers')
    st.markdown("Data is already cleaned and each row is an accident record. No outliers needed")
    
    
    # Inspect the dataset
    st.markdown('### Inspect and visualize some interesting features')

    # Restore dataset if already in memory
    st.session_state['house_df'] = df

    # Display dataframe as table
    st.dataframe(df.describe())

    ###################### VISUALIZE DATASET #######################
    st.markdown('### Visualize Features')

    numeric_columns = list(df.select_dtypes(include='number').columns)
    #numeric_columns = list(df.select_dtypes(['float','int']).columns)    
    # Specify Input Parameters
    st.sidebar.header('Specify Input Parameters')

    # Collect user plot selection
    st.sidebar.header('Select type of chart')
    chart_select = st.sidebar.selectbox(
        label='Type of chart',
        options=['Scatterplots', 'Lineplots', 'Histogram', 'Boxplot']
    )

    # Draw plots
    if chart_select == 'Scatterplots':
        try:
            x_values = st.sidebar.selectbox('X axis', options=numeric_columns)
            y_values = st.sidebar.selectbox('Y axis', options=numeric_columns)
            side_bar_data = sidebar_filter(
                df, chart_select, x=x_values, y=y_values)
            plot = px.scatter(data_frame=df,
                              x=x_values, y=y_values,
                              range_x=[side_bar_data[0][0],
                                       side_bar_data[0][1]],
                              range_y=[side_bar_data[1][0],
                                       side_bar_data[1][1]])
            st.write(plot)
        except Exception as e:
            print(e)
    if chart_select == 'Histogram':
        try:
            x_values = st.sidebar.selectbox('X axis', options=numeric_columns)
            side_bar_data = sidebar_filter(df, chart_select, x=x_values)
            plot = px.histogram(data_frame=df,
                                x=x_values,
                                range_x=[side_bar_data[0][0],
                                         side_bar_data[0][1]])
            st.write(plot)
        except Exception as e:
            print(e)
    if chart_select == 'Lineplots':
        try:
            x_values = st.sidebar.selectbox('X axis', options=numeric_columns)
            y_values = st.sidebar.selectbox('Y axis', options=numeric_columns)
            side_bar_data = sidebar_filter(
                df, chart_select, x=x_values, y=y_values)
            plot = px.line(df,
                           x=x_values,
                           y=y_values,
                           range_x=[side_bar_data[0][0],
                                    side_bar_data[0][1]],
                           range_y=[side_bar_data[1][0],
                                    side_bar_data[1][1]])
            st.write(plot)
        except Exception as e:
            print(e)
    if chart_select == 'Boxplot':
        try:
            x_values = st.sidebar.selectbox('X axis', options=numeric_columns)
            side_bar_data = sidebar_filter(df, chart_select, x=x_values)
            plot = px.box(df,
                          x=x_values,
                          range_x=[side_bar_data[0][0],
                                   side_bar_data[0][1]])
            st.write(plot)
        except Exception as e:
            print(e)
    

    # Normalize your data if needed
    st.markdown('### Categorizing data')
    st.markdown("For each accidient, generate boolean variable on the combination of accident cause from both parties")
    df['combined'] = df['CONTRIBUTING FACTOR VEHICLE 1'].astype(str) + '_' + df['CONTRIBUTING FACTOR VEHICLE 2'].astype(str)
    dummies = pd.get_dummies(df['combined'])
    df = pd.concat([df, dummies], axis=1)
    df.drop('combined', axis=1, inplace=True)
    
    locations = pd.concat([dfl['LATITUDE'],dfl['LONGITUDE']],axis=1)
    df1 = pd.concat([dummies,locations],axis=1)
    
    #df['combined'] = df['NUMBER OF PERSONS INJURED'].astype(str) + '_' + df['NUMBER OF PERSONS KILLED'].astype(str)
    #dummies = pd.get_dummies(df['combined'])
    #df = pd.concat([df, dummies], axis=1)
    #df.drop('combined', axis=1, inplace=True)
    st.dataframe(df)
    
    st.map(dfl)
    if "update" not in st.session_state:
        
        st.session_state['update'] = df
    st.write('Continue to Train Model')
    pd.DataFrame(df).to_csv('df')
    df1.to_csv('df1')