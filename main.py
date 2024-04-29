import sys
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import joblib
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from PyQt6.QtWidgets import (QMainWindow, QApplication, QWidget, QLabel,
                             QCheckBox, QComboBox, QSlider, QSpinBox, QDockWidget, QVBoxLayout,
                             QHBoxLayout,
                             QTabWidget, QPushButton, QDial)
from PyQt6.QtGui import QIcon, QAction
from PyQt6.QtCore import Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_absolute_error
import warnings

warnings.filterwarnings("ignore")


def train_model_and_scaler(model, scaler, X, y):
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Standardize the features using the scaler
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train the model
    model.fit(X_train_scaled, y_train)

    # Evaluate the model on the test set
    y_pred = model.predict(X_test_scaled)
    mae = mean_absolute_error(y_test, y_pred)

    print(f"Mean Absolute Error with test data: {mae}")

    return model, scaler


# Load your dataset
df = pd.read_csv('final_data_electricity.csv')

# 'amount_paid' is the target variable
X = df.drop('amount_paid', axis=1)  # Features
y = df['amount_paid']  # Target variable

# Create model and scaler instances
model = LinearRegression()
scaler = StandardScaler()

# Train the model and scaler
trained_model, trained_scaler = train_model_and_scaler(model, scaler, X, y)

# Save the trained model and scaler parameters to files
joblib.dump(trained_model, 'model.joblib')
joblib.dump(trained_scaler, 'scaler.joblib')


class Canvas(FigureCanvas):
    def __init__(self, parent=None):
        self.fig, self.ax = plt.subplots()
        super().__init__(self.fig)

    def plot_data(self, df, predicted_value=None):
        self.ax.clear()  # Clear the previous plot

        # Filter out data points with house area smaller than 400 and amount_paid higher than 60
        filtered_df = df[(df['housearea'] >= 400) | (df['amount_paid'] <= 60)]

        # Create a new DataFrame with summarized values for every 20 units of house area
        summarized_df = filtered_df.groupby(filtered_df['housearea'] // 20 * 20).agg(
            {'amount_paid': 'mean'}).reset_index()

        # Scatter plot with smaller and less visible dots
        sns.scatterplot(data=filtered_df, x='housearea', y='amount_paid', s=10, color='blue', marker='o', alpha=0.5,
                        label='Data Points', ax=self.ax)

        # Line plot
        sns.lineplot(data=summarized_df, x='housearea', y='amount_paid', color='red', label='Trend Line', ax=self.ax)

        # Plot the predicted value as a star
        if predicted_value is not None:
            self.ax.plot(predicted_value['housearea'], predicted_value['amount_paid'], marker='*', markersize=15,
                         color='red', markeredgecolor='black', markeredgewidth=1.5, label='Predicted Value')

        self.ax.set_xlabel('House Area')
        self.ax.set_ylabel('Amount Paid')
        self.ax.legend()  # Add legend

        self.draw()


class MplCanvas(FigureCanvas):
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        self.fig, self.ax = plt.subplots(figsize=(width, height), dpi=dpi)
        super().__init__(self.fig)

    def plot_histogram(self, data, title):
        self.fig.clf()
        ax = self.fig.add_subplot()  # Create a new set of axes

        # Plot the histogram
        data.hist(ax=ax, bins=20, color='blue', alpha=0.7)
        ax.set_title(title)
        ax.set_xlabel('Values')
        ax.set_ylabel('Frequency')
        self.fig.tight_layout()  # make sure they don't overlap

        self.draw()


class SecondWindow(QMainWindow):
    def __init__(self, data):
        super().__init__()

        self.setWindowTitle('Histograms')
        self.setGeometry(310, 110, 600, 600)

        wid = QWidget()
        layout2 = QVBoxLayout()
        self.chart2 = MplCanvas(self)
        self.chart2.plot_histogram(data, title="title")
        layout2.addWidget(self.chart2)
        self.setCentralWidget(wid)
        wid.setLayout(layout2)


class MainWindow(QMainWindow):

    def __init__(self):
        super().__init__()

        self.setWindowTitle('Electricity bill prediction')
        self.setGeometry(300, 100, 850, 500)

        self.initUI()
        self.show()

    def showSecondWindow(self):
        if self.win2.isHidden():
            self.win2.show()

    def initUI(self):

        self.win2 = SecondWindow(data=df)

        # INPUTS TAB

        self.inputRoom = QComboBox()
        self.inputRoom.setPlaceholderText("Select...")
        for i in range(6):
            self.inputRoom.addItem(str(i))
        self.inputRoom.currentIndexChanged.connect(self.updateValueRoom)

        self.inputPeople = QSlider(Qt.Orientation.Horizontal)
        self.inputPeople.setMaximum(11)
        self.inputPeople.setMinimum(0)
        self.inputPeople.sliderPosition()
        self.inputPeople.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.inputPeople.valueChanged.connect(self.updateValuePeople)
        self.valuePeople = QLabel('0')

        self.inputChildren = QSpinBox()
        self.inputChildren.setMinimum(0)
        self.inputChildren.setMaximum(4)
        self.inputChildren.valueChanged.connect(self.updateValueChildren)

        self.inputArea = QDial()
        self.inputArea.setMinimum(244)  # min area
        self.inputArea.setMaximum(1189)  # max of area
        self.inputArea.valueChanged.connect(self.updateValueArea)
        self.valueArea = QLabel('244')

        self.inputUrban = QCheckBox()
        self.inputUrban.stateChanged.connect(self.updateValueUrban)

        self.inputFlat = QCheckBox()
        self.inputFlat.stateChanged.connect(self.updateValueFlat)

        self.inputAC = QCheckBox()
        self.inputAC.stateChanged.connect(self.updateValueAC)

        self.inputTV = QCheckBox()
        self.inputTV.stateChanged.connect(self.updateValueTV)

        slider_layout = QHBoxLayout()
        slider_layout.addWidget(self.inputPeople)
        slider_layout.addWidget(self.valuePeople)

        qdial_layout = QHBoxLayout()
        qdial_layout.addWidget(self.inputArea)
        qdial_layout.addWidget(self.valueArea)

        checkbox_layout = QHBoxLayout()
        checkbox_layout.addWidget(QLabel("Urban Area"))
        checkbox_layout.addWidget(self.inputUrban)

        checkbox_layout2 = QHBoxLayout()
        checkbox_layout2.addWidget(QLabel("Flat"))
        checkbox_layout2.addWidget(self.inputFlat)

        checkbox_layout3 = QHBoxLayout()
        checkbox_layout3.addWidget(QLabel("AC"))
        checkbox_layout3.addWidget(self.inputAC)

        checkbox_layout4 = QHBoxLayout()
        checkbox_layout4.addWidget(QLabel("TV"))
        checkbox_layout4.addWidget(self.inputTV)

        tabL = QVBoxLayout()

        # TAB WIDGETS
        tabL.addWidget(QLabel("Number of Rooms"))
        tabL.addWidget(self.inputRoom)
        tabL.addSpacing(5)

        tabL.addWidget(QLabel("Number of People"))
        tabL.addLayout(slider_layout)
        tabL.addSpacing(5)

        tabL.addWidget(QLabel("Number of Children"))
        tabL.addWidget(self.inputChildren)
        tabL.addSpacing(5)

        tabL.addWidget(QLabel("House Area"))
        tabL.addSpacing(5)
        tabL.addLayout(qdial_layout)
        tabL.addSpacing(5)

        tabL.addLayout(checkbox_layout)
        tabL.addLayout(checkbox_layout2)
        tabL.addLayout(checkbox_layout3)
        tabL.addLayout(checkbox_layout4)
        tabL.addSpacing(5)

        tabL.addStretch()

        # TabWidget
        tab1 = QWidget()

        tab1.setLayout(tabL)

        tabWidget = QTabWidget()
        tabWidget.addTab(tab1, 'Input')

        innerDockWidget = QWidget()
        innerDockWidget.setFixedSize(200, 500)

        outerTabWidLay = QVBoxLayout()
        outerTabWidLay.addWidget(tabWidget)
        innerDockWidget.setLayout(outerTabWidLay)

        # dockWidget
        dockWidget = QDockWidget()
        dockWidget.setFeatures(QDockWidget.DockWidgetFeature.NoDockWidgetFeatures)
        dockWidget.setWidget(innerDockWidget)

        # SECOND TAB WITH LABELS

        self.lblRooms = QLabel('Number of Rooms:  ')
        self.lblPeople = QLabel('Number of People:  ')
        self.lblChildren = QLabel('Number of Children:  ')

        self.lblAC = QLabel('AC: No')
        self.lblTV = QLabel('TV: No')

        self.lblArea = QLabel('House Area:  ')
        self.lblUrban = QLabel('Urban Area: No')
        self.lblFlat = QLabel('Flat: No')

        colLayout1 = QVBoxLayout()
        colLayout2 = QVBoxLayout()
        colLayout3 = QVBoxLayout()
        colLayout4 = QVBoxLayout()

        # 1st column
        colLayout1.addWidget(self.lblRooms)
        colLayout1.addWidget(self.lblPeople)
        colLayout1.addWidget(self.lblChildren)

        # 2nd column
        colLayout2.addWidget(self.lblArea)
        colLayout2.addWidget(self.lblUrban)
        colLayout2.addWidget(self.lblFlat)

        # 3rd column
        colLayout3.addWidget(self.lblAC)
        colLayout3.addWidget(self.lblTV)

        # predict push-button
        predictButton = QPushButton("Predict")
        predictButton.clicked.connect(self.predictAmountPaid)
        self.predictionLabel = QLabel('')

        # Layout

        # 4th column
        colLayout4.addWidget(QWidget())
        colLayout4.addWidget(predictButton)
        colLayout4.addWidget(self.predictionLabel)

        outerLblLay = QHBoxLayout()
        outerLblLay.addLayout(colLayout1)
        outerLblLay.addSpacing(20)
        outerLblLay.addLayout(colLayout2)
        outerLblLay.addSpacing(20)
        outerLblLay.addLayout(colLayout3)
        outerLblLay.addSpacing(20)
        outerLblLay.addLayout(colLayout4)

        centWidget = QWidget()

        centLay = QVBoxLayout()
        self.chart = Canvas(self)

        centLay.addWidget(self.chart, stretch=4)
        self.chart.plot_data(df)

        centLay.addLayout(outerLblLay, stretch=1)
        centWidget.setLayout(centLay)

        self.setCentralWidget(centWidget)
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, dockWidget)

        # Show histograms in window 2 "file > histogram"
        showHist = QAction("Histogram", self)
        showHist.triggered.connect(self.showSecondWindow)
        second_window = SecondWindow(df)
        second_window.show()

        # menubar
        menubar = self.menuBar()
        fileMenu = menubar.addMenu('&File')
        fileMenu.addAction(showHist)

    def updateValueRoom(self, value):
        room_text = f'Number of Rooms: {value}'
        self.lblRooms.setText(room_text)

    def updateValuePeople(self, value):
        self.valuePeople.setText(str(value))
        self.lblPeople.setText(f'Number of People: {value}')

    def updateValueChildren(self, value):
        self.lblChildren.setText(f'Number of Children: {value}')

    def updateValueArea(self, value):
        self.valueArea.setText(str(value))
        self.lblArea.setText(f'House Area: {value}')

    def updateValueUrban(self, state):
        urban_text = 'Urban Area: Yes' if state == 2 else 'Urban Area: No'
        self.lblUrban.setText(urban_text)

    def updateValueFlat(self, state):
        flat_text = 'Flat: Yes' if state == 2 else 'Flat: No'
        self.lblFlat.setText(flat_text)

    def updateValueAC(self, state):
        ac_text = 'AC: Yes' if state == 2 else 'AC: No'
        self.lblAC.setText(ac_text)

    def updateValueTV(self, state):
        tv_text = 'TV: Yes' if state == 2 else 'TV: No'
        self.lblTV.setText(tv_text)

    def predictAmountPaid(self):
        try:
            # Load the trained model and scaler
            loaded_model = joblib.load('model.joblib')
            loaded_scaler = joblib.load('scaler.joblib')

            # Collect user input
            user_input = {
                'num_rooms': self.inputRoom.currentText() if self.inputRoom.currentText() else 0,
                'num_people': self.inputPeople.value(),
                'num_children': self.inputChildren.value(),
                'housearea': max(self.inputArea.value(), 244),  # Ensure housearea is at least 244
                'is_urban': self.inputUrban.isChecked(),
                'is_flat': self.inputFlat.isChecked(),
                'is_ac': self.inputAC.isChecked(),
                'is_tv': self.inputTV.isChecked()
            }

            # Create a DataFrame with the user input
            user_input_df = pd.DataFrame([user_input])

            # Ensure columns order is the same as during training
            user_input_df = user_input_df[X.columns]

            # Scale user input using the loaded scaler
            user_input_scaled = loaded_scaler.transform(user_input_df)

            # Make a prediction using the loaded model
            prediction = loaded_model.predict(user_input_scaled)
            formatted_prediction = 'Predicted Amount: ' + format(prediction[0], '.2f') + ' EUR'

            # Calculate MAE on a sample
            test_set = pd.read_csv('final_data_electricity.csv')
            test_X = test_set.drop('amount_paid', axis=1)
            test_y = test_set['amount_paid']
            test_input_scaled = loaded_scaler.transform(test_X)
            test_prediction = loaded_model.predict(test_input_scaled)
            mae = mean_absolute_error(test_y, test_prediction)
            print(f'Mean Absolute Error on Full Data: {mae}')

            # Display the predicted amount_paid and MAE
            self.showPredictionLabel(formatted_prediction + f'\nMean Error: {mae:.2f}')
            # Update the plot with the predicted value
            predicted_value = {'housearea': user_input['housearea'], 'amount_paid': prediction[0]}
            self.chart.plot_data(df, predicted_value)

            # Display the predicted amount_paid (you can replace print with updating a QLabel or similar)
            print("Predicted Amount Paid:", format(prediction[0], '.2f'))

        except Exception as e:
            # Display the error message
            print("Error:", str(e))

    def showPredictionLabel(self, text):
        # Display the predicted amount_paid
        self.predictionLabel.setText(text)

        # Change text color to red
        self.predictionLabel.setStyleSheet("color: red;")

        # Make text bold
        font = self.predictionLabel.font()
        font.setBold(True)
        self.predictionLabel.setFont(font)


def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    sys.exit(app.exec())


if __name__ == '__main__':
    main()
