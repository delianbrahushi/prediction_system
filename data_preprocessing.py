import pandas as pd

class YourClass:
    def __init__(self):
        self.electricity_data = None  # Initialize the variable to store the loaded data

    def loadData(self, file_path='household_energy_bill_data.csv'):
        # Read the CSV file into a Pandas DataFrame
        try:
            self.electricity_data = pd.read_csv(file_path)
        except FileNotFoundError:
            print(f"File '{file_path}' not found.")
            return
        except pd.errors.EmptyDataError:
            print(f"File '{file_path}' is empty.")
            return

        # Drop the column 'ave_monthly_income' if it exists
        if 'ave_monthly_income' in self.electricity_data.columns:
            self.electricity_data = self.electricity_data.drop(columns=['ave_monthly_income'])

        # Replace all "-1" values with 0 in columns 'num_rooms' and 'num_people'
        self.electricity_data[["num_rooms", "num_people"]] = self.electricity_data[["num_rooms", "num_people"]].replace(-1, 0)

        # Convert the prices in the "amount_paid" column from INR to EUR and round to 2 decimal places
        conversion_rate = 0.11038
        self.electricity_data['amount_paid'] = self.electricity_data['amount_paid'].apply(lambda x: round(x * conversion_rate, 2))

        # Save the modified DataFrame to a new CSV file named 'final_data_electricity.csv'
        self.electricity_data.to_csv('final_data_electricity.csv', index=False)

        # Print the number of rows
        num_rows = len(self.electricity_data)  # or self.electricity_data.shape[0]
        print(f"Number of rows: {num_rows}")

# Usage
your_instance = YourClass()
your_instance.loadData()
