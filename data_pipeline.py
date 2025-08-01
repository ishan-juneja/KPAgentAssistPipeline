#from src.data.data_load import load_and_merge_csvs
from src.data.data_clean import find_common_phrases
#from src.data.data_clean import label_ngrams
#from src.data.data_clean import filter_phrases
from src.data.data_clean import filter_phrases_by_noise
from src.data.data_clean import shorten_summary
from src.data.data_add import preprocess_conversation_length_dataframe
from src.data.data_add import preprocess_conversation_times_dataframe
from src.models.label_model import TaxonomyLabeler
import pandas as pd

def finalize_data():
    # print("Running the data pipeline!")
    
    # # We run this if given a new CSV and add it to our frame
    # # Maybe consider in the future preprocessing it and then adding it to the dataframe
    # # df = load_and_merge_csvs("data/raw/Agent_Assist_Data")

    # df = pd.read_csv("data/processed/cleaned_feedback.csv")

    # print("Columns in DataFrame:", df.columns.tolist())
    
    #df = filter_phrases(df)
    print("preprocessing")

    #Duplicating the Knowledge_Answer column as this gets edited later
    # df["Original_Knowledge_Answer"] = df["Knowledge_Answer"]

    # df = filter_phrases_by_noise(df)
    # print("finished filtering by noise")
    # df = preprocess_conversation_times_dataframe(df)
    # print("finished adding time")
    # df = preprocess_conversation_length_dataframe(df)
    # print("finished adding length")
    # df = shorten_summary(df)
    # print("finished shortening summary")

    # df.to_csv("data/processed/final_cleaned_feedback_2.csv", index=False)

    # Now we will be labeling our dataset based on the error taxonomy
    

    # print("finding common phrases")
    # phrases = find_common_phrases(df)
    # pd.set_option("display.max_rows", 500)
    # print(phrases.head(500))
