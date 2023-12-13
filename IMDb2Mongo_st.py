#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import os
import dns
import copy
import json
import streamlit as st
import dask.dataframe as dd
from dask.diagnostics import ProgressBar
from multiprocessing import cpu_count
import pymongo
import warnings
warnings.filterwarnings("ignore")
import imdb
from imdb import IMDb, IMDbError
from datetime import datetime
ia = IMDb()

def read_table():
    """
    Prompt the user to enter the path of a CSV file.

    This function asks the user to input the file path of a CSV file. If no input is
    provided, it raises a SystemExit exception, indicating that no filename was supplied.

    Returns:
        str: The path of the CSV file entered by the user.
    """
    
    # Prompt the user to enter the file path
    print("Dataset to read")
    filename = input("Please enter the path of the CSV file: ")

    # Check if the filename is provided
    if not filename:
        # Inform the user and exit if no filename is provided
        print("No filename supplied, exit")
        raise SystemExit("Cancelling: no filename supplied")

    return filename

def get_infoset():
    """
    Prompt the user to select additional infosets from a list.

    The function retrieves a list of available infosets, excludes certain unwanted infosets,
    and then asks the user to select from the remaining infosets. The selected infosets are
    then returned.

    Returns:
        list: A list of selected infosets chosen by the user.
    """

    # Get a list of movie infosets and exclude unwanted ones
    infolist = ia.get_movie_infoset()
    unwanted_infosets = {'main', 'news', 'soundtrack'}
    infolist = [ele for ele in infolist if ele not in unwanted_infosets]

    # Display available infosets and instructions for selection
    print("The default infoset is 'main'. You can add more infosets.")
    for i, info in enumerate(infolist, 1):
        print(f"{i}. {info}")

    # Prompt the user to enter their choice of infosets
    selected_indices = input("Enter the numbers of the infosets you wish to add, separated by commas: ")

    # Process the input and generate the list of selected infosets
    selected_infosets = selected_indices.split(',')
    selected_infosets = [infolist[int(index) - 1] for index in selected_infosets if index.isdigit()]

    # Display the selected infosets
    strv = ", ".join(['main'] + selected_infosets)
    print(f"Chosen infosets: {strv}")

    return selected_infosets

def get_database():
    """
    Prompt the user to enter details for database connection.

    This function asks the user to enter the connection string, database name, and
    collection name for a database connection. These values are then returned in a tuple.

    Returns:
        tuple: A tuple containing the connection string, database name, and collection name.
    """
    
    # Prompt the user for database connection details
    print("Please enter MongoDB connection string, Database and Collection names")
    connection_string = input("Connection string: ")
    database_name = input("Database: ")
    collection_name = input("Collection: ")

    # Return the collected values
    return connection_string, database_name, collection_name

def get_data(filename):
    """
    Reads the list of titles from the file, deletes the first row if it's not a typical IMDb code, 
    and deletes the first two letters of the code.

    Args:
        filename (str): The path to the CSV file to be read.

    Returns:
        DataFrame: A pandas DataFrame with the modified '_id' column.

    Raises:
        SystemExit: If the dataset is incorrect or the file cannot be read.
    """
    try:
        # Read the CSV file without assuming first row is header
        titles = pd.read_csv(filename, header=None, names=['_id'])

        # Check if the first row is a typical IMDb code
        if not titles.iloc[0, 0].startswith('tt'):
            # If not, drop the first row
            titles = titles.drop(titles.index[0])

        # Delete the first two letters of the code in the '_id' column
        titles['_id'] = titles['_id'].str.slice_replace(start=0, stop=2, repl='')

    except Exception as e:
        # Print error message and exit
        print(f"The dataset is incorrect or cannot be read. Error: {e}")
        raise SystemExit("Cancelling: The dataset is incorrect or cannot be read")

    return titles



def identify(DataObj):
    """
    Identifies and tags an IMDb object with a specific prefix based on its type.

    Args:
        DataObj: An IMDb object which can be of type Person, Movie, or Company.

    Returns:
        dict: A dictionary with a key '_id' and a value that is a combination of a tag
              based on the object's type and its IMDb ID.

    Raises:
        TypeError: If DataObj is not an instance of a recognized IMDb object type.
    """
    idoc = {}
    tag = ''

    # Assign a tag based on the type of the IMDb object
    if isinstance(DataObj, imdb.Person.Person):
        tag = 'nm'
    elif isinstance(DataObj, imdb.Movie.Movie):
        tag = 'tt'
    elif isinstance(DataObj, imdb.Company.Company):
        tag = 'co'
    else:
        # Raise an exception if DataObj is not a recognized type
        raise TypeError("Unrecognized IMDb object type.")

    # Combine the tag with the object's ID
    ID = DataObj.getID()
    idoc['_id'] = tag + str(ID)

    return idoc


def convert(DataObj):
    """
    Converts an IMDb object into a dictionary format, handling nested structures.

    Args:
        DataObj: An IMDb object, which can contain nested structures like dictionaries,
                 lists, or specific IMDb object types (Person, Movie, Company).

    Returns:
        dict: A dictionary representation of the IMDb object.
    """
    document = {}
    classes = (imdb.Person.Person, imdb.Movie.Movie, imdb.Company.Company)

    for key in DataObj.keys():
        # Handle nested dictionaries
        if isinstance(DataObj[key], dict):
            document[key] = convert(DataObj[key])

        # Handle lists
        elif isinstance(DataObj[key], list):
            document.update(identify(DataObj))
            values = DataObj[key]

            if len(values) == 0:
                continue

            sample = values[0]

            # Process lists of specific IMDb object types
            if isinstance(sample, classes):
                val = [x.data for x in values]
                for x in val:
                    n = val.index(x)
                    x.update(identify(values[n]))
                document[key] = val

            # Handle single-element lists
            elif len(values) == 1:
                if isinstance(values[0], classes):
                    data = values[0].data
                    data.update(identify(values[0]))
                    document[key] = [data]
                else:
                    document[key] = values[0]

            # Process lists of strings or bytes
            elif isinstance(sample, (str, bytes)):
                document[key] = DataObj[key]

        # Handle specific IMDb object types
        elif isinstance(DataObj[key], classes):
            data = DataObj[key].data
            data.update(identify(DataObj[key]))
            document[key] = convert(data)

        # Handle other types
        else:
            document[key] = DataObj[key]

    return document


def append_error_message(error_message, max_file_size=1048576):  # 1 MB di dimensione massima per default
    """
    Append an error message with a timestamp to the file 'errors.txt'. Implements log rotation.

    Args:
        error_message (str): The error message to be appended to the file.
        max_file_size (int): The maximum size of the log file in bytes before rotation (default is 1 MB).
    """
    log_file = 'errors.txt'
    rotated_log_file = 'errors_old.txt'

    # Check if the file exists and its size
    if os.path.isfile(log_file) and os.path.getsize(log_file) > max_file_size:
        # Rotate the log file
        if os.path.isfile(rotated_log_file):
            os.remove(rotated_log_file)
        os.rename(log_file, rotated_log_file)

    with open(log_file, "a+") as file_object:
        # Append the timestamp and the error message
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        file_object.write(f"{timestamp} - {error_message}\n")


def get_main(title, infoset):
    """
    Download the filmography information for a given title from IMDb and convert it to JSON format.

    Args:
        title (str): The IMDb title identifier (e.g., '0133093' for 'The Matrix').
        infoset (list): A list of information sets to retrieve (e.g., ['main', 'episodes']).

    Returns:
        str: A JSON string containing the requested information, or None if an error occurs.
    """
    try:
        # Attempt to retrieve the movie information
        mv = ia.get_movie(title, info=infoset)
    except IMDbError as e:
        # Log the error and return None
        append_error_message(f"Error retrieving movie with title {title}: {str(e)}")
        return None

    # Convert the movie information to JSON
    movie = json.dumps(convert(mv))
    return movie

def dask_impl(df, infoset):
    """
    Applies the 'get_main' function in parallel to each title identifier in the DataFrame.

    Args:
        df (DataFrame): A pandas DataFrame containing title identifiers.
        infoset (list): A list of information sets to retrieve for each title.

    Returns:
        Dask DataFrame: A DataFrame with results from the 'get_main' function.
    """
    # Number of available CPU cores
    CPU_COUNT = cpu_count()

    # Initialize the progress bar
    ProgressBar().register()

    # Convert the pandas DataFrame to a Dask DataFrame and apply 'get_main' in parallel
    return dd.from_pandas(df, npartitions=CPU_COUNT).apply(
        lambda row: get_main(row['_id'], infoset),
        axis=1,
        meta='str'  # Adjust based on the expected return type of 'get_main'
    ).compute()

def apply_impl(df, infoset):
    """
    Apply the 'get_main' function to each row of the pandas DataFrame in a non-parallel manner.

    Args:
        df (DataFrame): The pandas DataFrame containing title identifiers.
        infoset (list): The list of information sets to retrieve for each title.

    Returns:
        Series/DataFrame: The result of applying 'get_main' to each row of df.
    """
    return df.apply(lambda row: get_main(row['_id'], infoset), axis=1)

def connect(values):
    """
    Establishes a connection to a MongoDB collection using a list of parameters.

    Args:
        values (list): A list containing the MongoDB connection URI, database name, and collection name.

    Returns:
        Collection: A pymongo collection object.

    Raises:
        ValueError: If the 'values' list does not contain the required elements.
        ConnectionError: If the connection to the MongoDB server fails.
    """
    if len(values) < 3:
        raise ValueError("Insufficient connection parameters provided.")

    connection_string, database_name, collection_name = values
    try:
        client = pymongo.MongoClient(connection_string)
        db = client[database_name]
        collection = db[collection_name]
        return collection
    except pymongo.errors.ConnectionFailure as e:
        raise ConnectionError(f"Failed to connect to MongoDB: {e}")

def to_mongo(mov, values):
    """
    Inserts a JSON movie record into a MongoDB collection.

    Args:
        mov (str): A string representation of a movie record in JSON format.
        values (list): A list containing the MongoDB connection URI, database name, and collection name.

    Raises:
        ValueError: If 'mov' cannot be parsed as JSON.
        pymongo.errors.ConnectionFailure: If the connection to the MongoDB server fails.
        pymongo.errors.PyMongoError: If an error occurs during the insertion.
    """
    try:
        collection = connect(values)
        pyresponse = json.loads(mov)
        collection.insert_one(pyresponse)
    except json.JSONDecodeError:
        raise ValueError("Invalid JSON format.")
    except pymongo.errors.ConnectionFailure as e:
        raise pymongo.errors.ConnectionFailure(f"Failed to connect to MongoDB: {e}")
    except pymongo.errors.PyMongoError as e:
        raise pymongo.errors.PyMongoError(f"Error inserting data into MongoDB: {e}")

def app(df, values):
    """
    Apply the 'to_mongo' function to each row of the DataFrame in parallel.

    This function processes a pandas DataFrame in parallel, applying the
    'to_mongo' function to each row. It handles any exceptions that occur
    during the processing.

    Args:
        df (DataFrame): The pandas DataFrame to process.
        values (list): The list containing connection details for MongoDB.

    Raises:
        SystemExit: If an error occurs during processing.
    """
    try:
        # Process the DataFrame in parallel
        dd.from_pandas(df, npartitions=cpu_count()).apply(
            to_mongo, args=(values,), meta=(int)).compute()
    except Exception as e:
        # Print the error message and exit
        print(f"An error occurred: {e}")
        raise SystemExit(f"Cancelling: {e}")

def main():
    """
    Main function to process and upload movie data.

    - Reads a filename for movie data.
    - Retrieves titles from the specified file.
    - Gathers database connection details.
    - Obtains additional information set preferences.
    - Processes the movie data in parallel and uploads it to the specified MongoDB database.
    - Notifies the user upon successful completion of the operation.
    """

    st.title("IMDb to MongoDB")

    # Input for the CSV file path
    filename = st.file_uploader("Upload your CSV file", type=["csv"])

    # Selection of infosets
    all_infosets = ia.get_movie_infoset()
    unwanted_infosets = {'main', 'news', 'soundtrack'}
    available_infosets = [info for info in all_infosets if info not in unwanted_infosets]
    # Multiselect for infosets excluding 'main'
    selected_infosets = st.multiselect("Select additional infosets to include:", available_infosets)

    # Always include 'main' in the final selection
    infoset = ['main'] + selected_infosets

    # Database connection details
    st.subheader("Database Connection Details")
    connection_string = st.text_input("Connection string:")
    database_name = st.text_input("Database name:")
    collection_name = st.text_input("Collection name:")
    
    if st.button("Process Data"):
        if filename and connection_string and database_name and collection_name:
            try:
                # Data processing
                titles = get_data(filename)
                values = (connection_string, database_name, collection_name)
                
                st.write("Processing data...")
                df = dask_impl(titles, infoset)
                df.dropna(inplace=True)
                app(df, values)

                st.success("Operation completed successfully")
            except Exception as e:
                st.error(f"An error occurred: {e}")
        else:
            st.warning("Please fill all the required fields")

if __name__ == "__main__":
    main()



