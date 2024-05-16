import streamlit as st                  # pip install streamlit
from helper_functions import fetch_dataset
import pandas as pd
import sklearn.model_selection
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import Lasso

#############################################

st.markdown("# Practical Applications of Machine Learning (PAML)")

#############################################

st.markdown("### Final Project - Pope - Ye")

#############################################

st.title('Train Model')

#############################################
df = None
df = pd.read_csv('df')
df1 = None
df1 = pd.read_csv('df1')
df = df1.drop('Unnamed: 0',axis=1)

df["Location_scale"] = (df1['LATITUDE']**2 )* (df1['LONGITUDE']**2)



if df is not None:
    # Display dataframe as table
    #st.dataframe(df)

    # Select variable to predict
    st.markdown('### Variable to predict')
    feature_predict_select = st.selectbox(
        label='Select Location_scale',
        options=df.columns,
        key='feature_selectbox',
    )

    # Select input features
    st.markdown('### Input features')
    feature_input_select = df1
    st.dataframe(df1)

    st.write('You selected output {}'.format(
        feature_predict_select))
    
    X = df1
    Y = df[feature_predict_select]
    st.dataframe(Y)
    # Split dataset
    st.markdown('### Split dataset into Train/Validation/Test sets')
    st.markdown(
        '#### Enter the percentage of validation/test data to use for training the model')
    number = st.number_input(
        label='Enter size of test set (X%)', min_value=0, max_value=100, value=30, step=1)
    full = pd.concat([Y,X],axis=1)
    XY_train,XY_test = sklearn.model_selection.train_test_split(full, test_size=number)
    
    XY_train = pd.DataFrame(XY_train)
    XY_test = pd.DataFrame(XY_test)
    
    
    
    Y_train = pd.DataFrame(XY_train)[feature_predict_select]
    X_train = XY_train.drop([feature_predict_select],axis=1)
    
    Y_test = XY_test[feature_predict_select]
    X_test = XY_test.drop([feature_predict_select],axis=1)
    X_train.to_csv('xtrain')
    X_test.to_csv('xtest')
    Y_train.to_csv('ytrain')
    Y_test.to_csv('ytest')
    # Train models
    st.markdown('### Train models')
    model_options = ['Lasso Regression',"Tree"]

    # Collect ML Models of interests
    model_select = st.multiselect(
        label='Select regression model for prediction',
        options=model_options,
    )
    st.write('You selected the follow models: {}'.format(model_select))
    
    
    
    if (model_options[0] in model_select):
        st.markdown('#### ' + model_options[0])

        param_col1, param_col2 = st.columns(2)
        with (param_col1):
            param1_options = []
            param1_select = st.selectbox(
                label='Select param1',
                options=param1_options,
                key='param1_select'
            )
            st.write('You select the following <param1>: {}'.format(param1_select))

            param2_options = []
            param2_select = st.selectbox(
                label='Select param2',
                options=param2_options,
                key='param2_select'
            )
            st.write('You select the following <param2>: {}'.format(param2_select))

        with (param_col2):
            param3_options = []
            param3_select = st.selectbox(
                label='Select param3',
                options=param3_options,
                key='param3_select'
            )
            st.write('You select the following <param3>: {}'.format(param3_select))

            param4_options = []
            param4_select = st.selectbox(
                label='Select param4',
                options=param4_options,
                key='param4_select'
            )
            st.write('You select the following <param4>: {}'.format(param4_select))

        model_params = {
            'param1': param1_select,
            'param2': param2_select,
            'param3': param3_select,
            'param4': param4_select
        }

        if st.button('Lasso Model'):
            model = Lasso(alpha=0.1, random_state=42)
            model.fit(X_train, Y_train)
            st.session_state['Lasso Regression'] = 1
            
            

        if model_options[0] not in st.session_state:
            st.write('Lasso Model is untrained')
        else:
            st.write('Lasso Model trained')
            st.write("Coefficients:",pd.concat([pd.Series(model.coef_)],axis=1))
            st.write(pd.Series(df1.columns))
            st.write(pd.concat([pd.Series(model.coef_),pd.Series(df1.columns)],axis=1))
        
        
    if (model_options[1] in model_select):
        st.markdown('#### ' + model_options[1])

        param_col1, param_col2 = st.columns(2)
        with (param_col1):
            param1_options = []
            param1_select = st.selectbox(
                label='Select param1',
                options=param1_options,
                key='param1_select'
            )
            st.write('You select the following <param1>: {}'.format(param1_select))

            param2_options = []
            param2_select = st.selectbox(
                label='Select param2',
                options=param2_options,
                key='param2_select'
            )
            st.write('You select the following <param2>: {}'.format(param2_select))

        with (param_col2):
            param3_options = []
            param3_select = st.selectbox(
                label='Select param3',
                options=param3_options,
                key='param3_select'
            )
            st.write('You select the following <param3>: {}'.format(param3_select))

            param4_options = []
            param4_select = st.selectbox(
                label='Select param4',
                options=param4_options,
                key='param4_select'
            )
            st.write('You select the following <param4>: {}'.format(param4_select))

        model_params = {
            'param1': param1_select,
            'param2': param2_select,
            'param3': param3_select,
            'param4': param4_select
        }

        
        if st.button('Tree'):
            model = DecisionTreeRegressor(max_depth=5, random_state=42)
            model.fit(X_train, Y_train)
            st.session_state['Tree'] = 1
            

        if model_options[1] not in st.session_state:
            st.write('Tree Model is untrained')
        else:
            st.write('Tree Model trained')
            st.write("Degrees of importance:",pd.concat([pd.Series(model.feature_importances_)],axis=1))
            #st.write(pd.Series(df1.columns))
            st.write(pd.concat([pd.Series(model.feature_importances_),pd.Series(df1.columns)],axis=1))
    
    import pickle
    filename = 'finalized_model.sav'
    pickle.dump(model, open(filename, 'wb'))