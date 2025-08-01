import pandas as pd
from src.data.data_pipeline import finalize_data
from src.analysis.analysis_error_finding import find_errors_parallel
from src.analysis.analysis_error_finding import find_errors_in_batches
from src.analysis.analysis_error_finding import develop_error_taxonomy
from src.analysis.analysis_organize import export_all_taxonomies_to_csv
from src.analysis.analysis_organize import label_sub_topics
from src.data.data_add import label_errors_with_taxonomy
from src.data.data_add import label_categories_with_taxonomy
# finalize_data()

print("Running!")
# develop_error_taxonomy()
# need to go to start_batch 12
#find_errors_in_batches(batch_size=20, start_batch=11, csv_store_file="llm_responses_5.csv") # This will also save it to csv
# label_sub_topics(csv_store_file="llm_responses_5.csv")

# df = pd.read_csv("my_findings.csv")
#label_errors_with_taxonomy(df)
# df = pd.read_csv("data/processed/final_cleaned_feedback_2.csv")
# print(df.head(5))
# label_categories_with_taxonomy(df=df, new_csv="final_dataset_july_7.csv")

# df = pd.read_csv("final_dataset_july_7.csv")
# label_errors_with_taxonomy(df=df, new_csv="final_dataset_july_7_2.csv")


print("unlabeled_df" in locals())