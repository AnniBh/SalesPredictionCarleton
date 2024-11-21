import streamlit as st
# import matplotlib.pyplot as plt
import numpy as np

# Set page configuration
st.set_page_config(page_title="Streamlit App", layout="wide")

# Sidebar Navigation
st.sidebar.title("Navigation")
options = st.sidebar.radio("Select a Page:", ["Home", "Visualization"])

# Main Section
if options == "Home":
    st.title("Welcome to the Streamlit App")
    st.write("This is a basic application with a fixed left navigation panel.")
    st.write("Use the navigation panel to switch between pages.")
    
elif options == "Visualization":
    st.title("Visualization Page")
    st.write("Here, we display a simple visualization.")
    
    # # Generate a simple plot
    # x = np.linspace(0, 10, 100)
    # y = np.sin(x)
    # fig, ax = plt.subplots()
    # ax.plot(x, y, label='sin(x)')
    # ax.set_title("Sine Wave")
    # ax.set_xlabel("X-axis")
    # ax.set_ylabel("Y-axis")
    # ax.legend()
    
    st.pyplot(fig)
