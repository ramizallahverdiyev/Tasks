import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from  PIL import Image
import os as os
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
import imblearn
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb


banner = Image.open('images/banner.png')
logo = Image.open('images/logo.png')
icon = Image.open('images/icon.png')

st.set_page_config(layout="wide", page_title="Week 12 CS", page_icon=icon)
st.title("Streamlit")

st.sidebar.image(banner, use_container_width='always')
menu = st.sidebar.selectbox("", ['Homepage', 'EDA', 'Modeling'])
dataset = st.selectbox("Choose a dataset", ['Loan Prediction', 'Water potability'])

st.text(dataset)
df = pd.read_csv(f'data/{dataset}.csv')

if dataset == 'Loan Prediction':
    df.drop("Loan_ID", axis=1, inplace=True)

if menu == "Homepage":
    st.dataframe(df)
    st.subheader("Data Describe")
    st.dataframe(df.describe())
    st.dataframe(df.isnull().sum())
    st.subheader("Data balance")
    st.dataframe(df.iloc[:, -1].value_counts())
    st.bar_chart(df.iloc[:, -1].value_counts())
    st.subheader("Features type")
    st.text(df.dtypes)
    st.title("BoxPlot")

    num_columns = df.select_dtypes("number").columns.values
    select = st.multiselect("Select columns", num_columns,num_columns)
    fig, ax = plt.subplots(figsize=(len(select) * 2, len(select) * 1.5))
    sns.boxplot(data=df[select], ax=ax, orient='h')

    ax.set_title(f"{dataset} outliers")
    ax.grid(True)
    sns.set_style("dark")
    sns.set_palette("pastel")
    st.pyplot(fig)

if menu == "EDA":
    def outlier(datacol):
        q1, q3 = df[datacol].quantile(0.25), df[datacol].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        df[datacol] = df[datacol].clip(lower = lower_bound, upper = upper_bound)
    
    for colname in df.select_dtypes(exclude="object").columns:
        outlier(colname)
    
    num_columns = df.select_dtypes("number").columns.values
    select = st.multiselect("Select columns", num_columns, num_columns)
    fig, ax = plt.subplots(figsize=(len(select) * 2, len(select) * 1.5))
    sns.boxplot(data=df[select], ax=ax, orient='h')

    ax.set_title(f"{dataset} outliers")
    ax.grid(True)
    sns.set_style("dark")
    st.pyplot(fig)

    df_nulls_nums = st.selectbox("Chosse fillna", ['Mode', 'Mean', 'Median'])

    if df_nulls_nums == 'Mode':
        for colname in df.columns:
            mode = df[colname].mode()
            df[colname].fillna(mode[0], inplace=True)

    if df_nulls_nums == 'Mean':
        for colname in df.select_dtypes(exclude="object").columns:
            mean = df[colname].mean()
            df[colname].fillna(mean, inplace=True)
    
    if df_nulls_nums == 'Median':
        for colname in df.columns:
            median = df[colname].median()
            df[colname].fillna(median, inplace=True)
    st.dataframe(df)

    df_encoder = st.selectbox("Choose fillna", ['LabelEncoder', 'OneHotEncoder'])

    if df_encoder == 'OneHotEncoder':
        df_dummies = pd.get_dummies(df.select_dtypes(include='object'))
    
    if df_encoder == 'LabelEncoder':
        from sklearn.preprocessing import LabelEncoder
        lb = LabelEncoder()
        for col in df.select_dtypes(include='object').columns:
            df[col] = lb.fit_transform(df[col])

    st.dataframe(df)

if menu == "Modeling":
    st.subheader("Train Test Split")
    X = df.drop(df.columns[-1], axis=1)
    y = df[df.columns[-1]]

    # X = pd.get_dummies(X)

    imputer = SimpleImputer(strategy='most_frequent')
    X = imputer.fit_transform(X)

    test_size = st.slider("Test size", min_value=0.1, max_value=0.5, value=0.3)
    random_state = st.slider("Random state", min_value=0, max_value=100, value=42)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    st.write(f"Training data shape: {X_train.shape}")
    st.write(f"Test data shape: {X_test.shape}")

    st.subheader("Scaling")
    scaling_option = st.selectbox("Choose scaling method", ['None','StandardScaler', 'MinMaxScaler', 'RobustScaler'])

    if scaling_option == 'StandardScaler':
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

    elif scaling_option == 'MinMaxScaler':
        scaler = MinMaxScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

    elif scaling_option == 'RobustScaler':
        scaler = RobustScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

    st.write("Data scaled using:", scaling_option)

    st.subheader("Choose model")
    model_option = st.selectbox("Model", ['Logistic Regression', 'Naive Bayes', 'K-Nearest Neighbors', 'Support Vector Machine', 'Random Forest', 'XGBoost'])

    if model_option == 'Logistic Regression':
        model = LogisticRegression(random_state=random_state)
    elif model_option == 'Naive Bayes':
        model = GaussianNB()
    elif model_option == 'K-Nearest Neighbors':
        n_neighbors = st.slider("Number of neighbors", min_value=1, max_value=20, value=5)
        model = KNeighborsClassifier(n_neighbors=n_neighbors)
    elif model_option == 'Support Vector Machine':
        kernel = st.selectbox("Kernel", ['linear', 'poly', 'rbf', 'sigmoid'])
        model = SVC(kernel=kernel, random_state=random_state)
    elif model_option == 'Random Forest':
        n_estimators = st.slider("Number of trees", min_value=10, max_value=200, value=100)
        model = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
    elif model_option == 'XGBoost':
        model = xgb.XGBClassifier(random_state=random_state)

    if st.button("Train Model"):
        model.fit(X_train, y_train)
        accuracy = model.score(X_test, y_test)
        st.write(f"Model accuracy: {accuracy:.2f}")

        st.subheader("Confusion Matrix")
        y_pred = model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)

        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
        ax.set_xlabel('Predicted labels')
        ax.set_ylabel('True labels')
        ax.set_title('Confusion Matrix')
        st.pyplot(fig)

        st.subheader("Classification Report")
        report = classification_report(y_test, y_pred, output_dict=True)
        report_df = pd.DataFrame(report).transpose()
        st.dataframe(report_df)