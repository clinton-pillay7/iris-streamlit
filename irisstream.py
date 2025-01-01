import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import streamlit as st
warnings.simplefilter("ignore")


def train_model():
    #Import iris dataset
    traindf=pd.read_csv(r'iris.csv')
    traindf=traindf.drop(columns="Id")
    x=traindf.iloc[:,:4]
    y=traindf.iloc[:,4]
    x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=0)
    model=LogisticRegression()
    model.fit(x_train,y_train)
    SepalLengthCm = st.number_input("Enter your SepalLengthCm:")
    SepalWidthCm = st.number_input("Enter your SepalWidthCm:")
    PetalLengthCm = st.number_input("Enter your PetalLengthCm:")
    PetalWidthCm = st.number_input("Enter your PetalwidthCm:")
    if st.button("Predict"):
        # Create a DataFrame
        data = [{"SepalLengthCm": SepalLengthCm, "SepalWidthCm": SepalWidthCm, "PetalLengthCm": PetalLengthCm , "PetalWidthCm": PetalWidthCm}]
        pred = pd.DataFrame(data)
        y_pred=model.predict(pred)
        st.write(y_pred)






if __name__ == "__main__":
    train_model()







