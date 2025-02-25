import os
import pandas as pd

# Specify the folder path and multiplication factor
folder_path = r"/Users/paulleonardwolff/Desktop/Stanford/Simulation_Python/Relevant_Files_Mujoco_trunk/trajectories/data_slow"  # Change this to your folder location
multiplication_factor = 1000  # Change this to your desired factor

def process_csv_files(folder, factor):
    # Ensure the folder exists
    if not os.path.exists(folder):
        print(f"Folder '{folder}' does not exist.")
        return

    # Define the new folder path with "_modified" appended
    parent_dir, folder_name = os.path.split(folder)
    modified_folder = os.path.join(parent_dir, f"{folder_name}_modified")

    # Create the modified folder if it doesn't exist
    os.makedirs(modified_folder, exist_ok=True)

    # Loop through all files in the folder
    for filename in os.listdir(folder):
        if filename.endswith(".csv"):
            file_path = os.path.join(folder, filename)
            try:
                # Read the CSV file
                df = pd.read_csv(file_path)

                # Ensure "time" column remains unchanged
                if "time" in df.columns:
                    df.loc[:, df.columns != "time"] = df.loc[:, df.columns != "time"].map(
                        lambda x: x * factor if isinstance(x, (int, float)) else x
                    )
                else:
                    df = df.map(lambda x: x * factor if isinstance(x, (int, float)) else x)

                # Save the modified file in the new folder with the same name
                new_file_path = os.path.join(modified_folder, filename)
                df.to_csv(new_file_path, index=False)

                print(f"Processed '{filename}' -> '{new_file_path}'")
            except Exception as e:
                print(f"Error processing '{filename}': {e}")

# Run the function
process_csv_files(folder_path, multiplication_factor)
