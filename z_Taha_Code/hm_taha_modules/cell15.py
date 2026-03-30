"""Auto-generated from HM_Taha.ipynb cell 15."""

#this code is to make any changes needed
import pandas as pd

def edit_cluster_data(file_path='i_refined_clusters_merged.csv'):
    """
    This program allows a user to edit data in a CSV file based on a ClusterID.
    """
    try:
        # Read the CSV file into a pandas DataFrame
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
        return

    while True:
        # Prompt the user for a ClusterID
        try:
            cluster_id_input = input("Please enter the ClusterID you want to edit (or 'q' to quit): ")
            if cluster_id_input.lower() == 'q':
                break

            cluster_id = int(cluster_id_input)

            # Find the row with the matching ClusterID
            if cluster_id not in df['ClusterID'].values:
                print(f"ClusterID {cluster_id} not found. Please try again.")
                continue

            # Get the index of the row to edit
            row_index = df[df['ClusterID'] == cluster_id].index[0]

            print(f"\nEditing data for ClusterID: {cluster_id}")
            print("="*30)

            # Iterate over each column for the selected row
            for col in df.columns:
                if col == 'ClusterID': # Don't allow editing of ClusterID
                    continue

                current_value = df.at[row_index, col]
                print(f"\nCurrent column: '{col}'")
                print(f"Current value: {current_value}")

                # Ask the user if they want to edit this column
                edit_choice = input(f"Do you want to edit the '{col}' column? (yes/no): ").lower()

                if edit_choice in ['yes', 'y']:
                    # Get the new value from the user
                    new_value = input(f"Enter the new value for '{col}': ")
                    df.at[row_index, col] = new_value
                    print(f"'{col}' has been updated.")
                else:
                    print(f"Skipping '{col}'.")

            # Save the updated DataFrame back to the CSV file
            df.to_csv(file_path, index=False)
            print("\n" + "="*30)
            print("All changes have been saved to the file.")
            print("="*30 + "\n")


        except ValueError:
            print("Invalid input. Please enter a numerical ClusterID.")
        except Exception as e:
            print(f"An error occurred: {e}")

if __name__ == "__main__":
    edit_cluster_data()
