import os
import pandas as pd


def load_and_merge_csvs(folder_path: str):
    # we need the folder path to access all of our files and go through those
    list_of_dfs = []


    for file in os.listdir(folder_path):
        try:
            # we now have all our csv within the folder

            # we now need to create a panda data frame out of the data
            file_path = os.path.join(folder_path, file)
            df = pd.read_csv(file_path) # reading our file
            df["Source_File"] = file

            # debugging column names
            print("File Name: " , file , "\n")
            print("Raw columns from file:", df.columns.tolist())

            # renaming columns to fit our scheme we are going for
            rename_map = {
                # This may be dropped fully as it offers no value
                #"Query": "Query", - our summary contains our query
                "Query / Topic": "Query_Type",
                "Reason": "Query_Type",

                "Feedback" : "Feedback",
                "Value" : "Feedback",

                "Topics" : "Conversation_Topic",
                "Sub-topics": "Coversation_Subtopic",

                "Type" : "Knowledge_Category",

                "Summary/Answer/Content": "Knowledge_Answer",

                "References" : "Knowledge",

                "Agent ID": "Agent_ID",
                "Agent NUID": "Agent_ID",

                "Created At": "Timestamp",

                "Comments": "Summary_Reason",
            }
            df = df.rename(columns=rename_map)
            # ANY OTHER COLUMNS THAT WANT TO BE ADDED SHOULD BE ADDED HERE

            required_cols = [
                "Query_Type", "Feedback", "Conversation_Topic", "Conversation_Subtopic", 
                "Knowledge_Answer", "Knowledge", "Agent_ID", "Timestamp", "Summary_Reason",
                "Source_File" # INSERT ANY OTHERS HERE
            ]

            # If a column isn't in our required columns it gets populated with None 
            # This however shouldn't happen considering our data integrity is all well
            for column in required_cols:
                if column not in df.columns:
                    print(column)
                    df[column] = None
            
            df = df[required_cols]

            # Add to our list
            list_of_dfs.append(df)
            
            # Debugging statements
            print(file)
            print(df.columns)
            print(df.head(10))
            print("\n")
        except Exception as error:
            print("Ran into the following error: " , error)

    merged_df = pd.concat(list_of_dfs, ignore_index=True) # combine all of our files, our indexing information
    # also isn't too useful here which is why we use ignore_index as True

    # after we combine them all into one big file we need to drop columns that aren't as relevant


    merged_df.drop_duplicates(inplace=True) # inplace allows us to not have to make a new data frame
    # df.drop()
    output_path = os.path.join("data", "processed", "cleaned_feedback.csv")
    merged_df.to_csv(output_path, index=False)


    print(merged_df.columns)
    print(merged_df.head(1000))




    return merged_df

def pre_clean_dataframe(df):
    """
    Standardizes raw CSV structure before renaming or merging.
    - Strips whitespace from column names
    - Replaces problematic characters (e.g., non-breaking spaces)
    - Removes duplicate header rows
    - Drops fully empty rows
    """
    df.columns = df.columns.str.strip().str.replace("\xa0", " ", regex=True)
    df = df.dropna(how="all")  # Drop rows where all values are NaN

    # convert positive and negative to 0s and 1s for training sake
    df["Feedback"] = df["Feedback"].toLowercase().map({"positive": 1, "negative" : 0})

    return df

