import os

# Example file path
file_path = "example/model.pkl"

# Extract the file extension using os.path.splitext()
file_base, file_extension = os.path.splitext(file_path)

# Print the results
print("File Base:", file_base)
print("File Extension:", file_extension)

# Store the file extension separately
stored_file_extension = file_extension  # Store the file extension in a variable or data structure

# Example usage:
if stored_file_extension == ".pkl":
    # Perform actions for pickle format
    print("This file has a .pkl extension.")

elif stored_file_extension == ".joblib":
    # Perform actions for joblib format
    print("This file has a .joblib extension.")

elif stored_file_extension == ".dll" or stored_file_extension == ".so":
    # Perform actions for DLL format
    print("This file has a DLL (.dll) or shared object (.so) extension.")

else:
    print("Unknown file extension.")
