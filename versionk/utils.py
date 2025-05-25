import os

def create_directory(folder_name):
    try:
        os.makedirs(folder_name)
        print(f"The directory '{folder_name}' was successfully created.")
    except FileExistsError:
        print(f"The directory '{folder_name}' already exists.")
    except Exception as e:
        print(f"Error creating directory '{folder_name}': {e}")