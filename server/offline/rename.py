import os

# Specify the directory where the files are located
directory = './straight-line-1/'

# Change to the directory
os.chdir(directory)

# Get a list of files in the directory
files = [file for file in os.listdir() if os.path.isfile(file) and file.startswith('frame')]

# Pad numbers with zeros and rename files
for file in files:
    # Extract the number from the file name

    number = int(file.split('_')[1].split('.')[0])
    print(number)
    # Pad the number with zeros
    padded_number = f"{number:04d}"

    # Create the new file name
    new_name = f"frame_{padded_number}.png"
    # Rename the file
    os.rename(file, new_name)