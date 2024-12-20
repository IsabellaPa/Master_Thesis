import os
import pandas as pd

# Specify the directory containing the Excel files
directory = 'C:/Users/ipatzke/OneDrive - Philips/Documents/Masterarbeit/MasterMission/Python_files/New_Master_Mission/Prototyp/Exel_sheets/Tests/Test_to_combine'

# Create a dictionary to store the file names and their corresponding DataFrames
file_dfs = {}

# Iterate through the Excel files in the directory
for filename in os.listdir(directory):
    if filename.endswith(".xlsx"):
        # Read the Excel file into a DataFrame
        df = pd.read_excel(os.path.join(directory, filename))
        
        # Create a new DataFrame with the desired structure
        new_df = pd.DataFrame({'File': [filename]})
        
        # Create a list to store the columns
        columns = []

        # Iterate through the categories and columns
        for category in df['Category']:
            for col in ['pre', 'intra']:
                col_name = f"{category} - {col}"
                value = df.loc[df['Category'] == category, col].values[0]
                columns.append(pd.Series([value], name=col_name))
        
        #  Add the columns to the DataFrame at once
        new_df = pd.concat([new_df] + columns, axis=1)
        
        # Add the new DataFrame to the dictionary
        file_dfs[filename] = new_df

# Combine the DataFrames into a single DataFrame
combined_df = pd.concat(file_dfs.values(), ignore_index=True)

# Write the combined DataFrame to a new Excel file
combined_df.to_excel(directory+'/combined_excel.xlsx', index=False)
print('done')