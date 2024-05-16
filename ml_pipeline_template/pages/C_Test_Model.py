import streamlit as st                  # pip install streamlit
from helper_functions import fetch_dataset
import pandas as pd
#############################################

st.markdown("# Practical Applications of Machine Learning (PAML)")

#############################################

st.markdown("### Final Project - Pope - Ye")
st.markdown("### Lasso to handle sparse matrix, Tree to find out decision based pattern")
#############################################

st.title('Test Model')

#############################################

df = None
df = fetch_dataset()
X_train = pd.read_csv('xtrain') 
X_test =pd.read_csv('xtest')
Y_train=pd.read_csv('ytrain')
Y_test=pd.read_csv('ytest')

if df is not None:
    st.markdown("### Get Performance Metrics")
    metric_options = ['placeholder']

    model_options = ['placeholder']

    trained_models = ["Lasso Regression","Tree"]

    # Select a trained classification model for evaluation
    model_select = st.multiselect(
        label='Select trained models for evaluation',
        options=trained_models
    )
    
    if (model_select):
        st.write(
            'You selected the following models for evaluation: {}'.format(model_select))

        eval_button = st.button('Evaluate your selected classification models')

        if eval_button:
            st.session_state['eval_button_clicked'] = eval_button
        import pickle
        model = pickle.load(open("finalized_model.sav", 'rb'))
        from sklearn.model_selection import cross_val_score

        scores = cross_val_score(model, X_test, Y_test, cv=5)
        st.write("Mean Accuracy using splitted test data:",scores.mean())
        st.write('Continue to Test Model')
        st.dataframe(df)
        if 'eval_button_clicked' in st.session_state and st.session_state['eval_button_clicked']:
            st.markdown('### Review Model Performance')

            review_options = ['plot', 'metrics']

            review_plot = st.multiselect(
                label='Select plot option(s)',
                options=review_options
            )

            if 'plot' in review_plot:
                pass

            if 'metrics' in review_plot:
                pass

    # Select a model to deploy from the trained models
    st.markdown("### Choose your Deployment Model")
    model_select = st.selectbox(
        label='Select the model you want to deploy',
        options=trained_models,
    )

    if (model_select):
        st.write('You selected the model: {}'.format(model_select))
        st.session_state['deploy_model'] = st.session_state[model_select]

    st.write('Continue to Deploy Model')
