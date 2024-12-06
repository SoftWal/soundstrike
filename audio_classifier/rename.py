import os

def rename_files_in_folder(folder_path):
    # Get the folder name from the path
    folder_name = os.path.basename(folder_path.rstrip(os.sep))
    
    # List all files in the folder
    files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
    
    # Iterate over the files and rename them
    for index, file_name in enumerate(files, start=1):
        # Get file extension
        file_extension = os.path.splitext(file_name)[1]
        # Create new file name
        new_name = f"{folder_name}_{index}{file_extension}"
        # Rename the file
        os.rename(os.path.join(folder_path, file_name), os.path.join(folder_path, new_name))
        print(f"Renamed: {file_name} -> {new_name}")

# Specify the folder path here
folder_path = r'C:\Users\Leona\Documents\UPRM\Capstone\audio_classifier\wavfiles\762x39'

# Run the renaming function
rename_files_in_folder(folder_path)
