Brahushi, Delian

 Electricity Bill Prediction

## Project description



**Electricity Bill Prediction App**

<img src="https://i.imgur.com/nPvOiGn.png" width="500" height="300" />
<img src="https://i.imgur.com/opb2sNk.png" width="300" height="300" />

This PyQt6 application provides a user-friendly interface for predicting electricity bills based on various input parameters. It incorporates data analysis using Pandas and NumPy, a Scikit-learn training model, and interactive visualization with Matplotlib.


## Installation

**Environment Details:**

- Python Version: 3.8.9
- pip Version: 23.3.1
- matplotlib: 3.7.4
- numpy: 1.24.4
- pandas: 2.0.3
- PyQt6: 6.6.1
- scikit-learn: 1.3.2
- My Operating System: macOS-12.3-arm64-arm-64bit (for context)

To install the Electricity Bill Prediction application, follow these steps:

1. Clone the project repository: 

`git clone https://mygit.th-deg.de/db27960/recommendation-system.git`

2. Navigate to the project directory:

 `cd electricity-bill-prediction`

3. Create and activate a virtual environment:

```
# On Windows
python -m venv venv
.\venv\Scripts\activate

# On macOS/Linux
python -m venv venv
source venv/bin/activate'
```
4. Install the required dependencies:

`pip install -r requirements.txt`


## Basic Usage

1. Run the data-preprocessing:

`python data_preprocessing.py`

2. Run the application:

`python main.py`

_**Note:**_ _The first time running the application may take up to 1 minute depending on the performance of the system._

3. Upon running the application, a window will appear, allowing you to input various features.

4. Enter the required information such as household details, and any other relevant features.

5. Press the **"Predict"** button to obtain the predicted electricity bill.



## Implementation of the Requests

The file `data_preprocessing.py` functions as a tool for refining noisy datasets. It excludes unnecessary columns, such as "ave_monthly_income" and adjusts values with no meaningful context from -1 to 0 in the 'num_rooms' and 'num_people' columns. Additionally, it converts the currency to EUR for enhanced clarity and convenience. The resulting dataset is saved as [final_data_electricity.csv](url) in the specified directory.

In the `main.py` file, which acts as the entry point for the Python code, there are two main sections. The first part involves designing the interface using PyQt6, while the second part focuses on generating a prediction model using scikit-learn. Upon the initial run, the prediction models are saved as `model.joblib` and `scaler.joblib`. All potentially confusing code segments are appropriately commented.

The `electricity_analysis.ipynb` file conducts Exploratory Data Analysis using Pandas, Seaborn, and NumPy. Presenting the data analysis in a Jupyter Notebook format is deemed more appealing than using the console. The notebook begins with an explanation of the data and its possible values. Subsequently, various analysis approaches are employed, including descriptive statistics, a correlation matrix, and various plots.


- The application is developed with PyQt6
- `requirements.txt` is provided, listing all necessary Python modules.
- The dataset was sourced from Kaggle. [LINK](https://www.kaggle.com/datasets/gireeshs/household-monthly-electricity-bill?select=Household+energy+bill+data.csv)
- Data importation is predefined.
- Data is analyzed with Pandas, NumPy and Seaborn in order to get an overview of the data. All of the Analysis is provided with a Jupyter Notebook file.
- The application features 8 input widgets: 4 QCheckBox, 1 QComboBox, 1 QSlider, 1 QSpinBox, and 1 QDial.
- The Linear Regression algorithm from scikit-learn is utilized for model training.
- Two output canvases for data visualization are included: one in the Main Window and another in the "Histogram" window. Histogram window provides 8 statistical metrics related to the input data.
- The app dynamically responds to changes in input parameters.




## Work done

Delian Brahushi has implemented all

## Copyright Claim

All components and content within this project have been implemented in accordance with ethical standards and best practices. The project has been developed with a commitment to integrity and compliance with relevant guidelines.

The dataset used is sourced from copyright-free website and obtained from [DATASET](https://www.kaggle.com/datasets/gireeshs/household-monthly-electricity-bill?select=Household+energy+bill+data.csv)

I take full disclosure of my work, emphasizing transparency and adherence to intellectual property rights.



