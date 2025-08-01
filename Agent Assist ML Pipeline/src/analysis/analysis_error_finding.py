import csv
import time
from src.models.zero_shot_LLM import prompt_llm
import pandas as pd
from src.analysis.analysis_organize import export_all_taxonomies_to_csv
import random, json
from concurrent.futures import ThreadPoolExecutor, as_completed
import os

derived_error_taxonomy = """
Agent-Related Issues:
1) Lack of Knowledge
2) Incomplete Information
3) Improper Escalation
4) Lack of Empathy
5) Rushed or Abrupt Closure
6) Failure to Follow Up

Process/Content Issues:
1) Overly Complex Instructions
2) Missing Documentation
3) Technical Jargon
4) Policy Ambiguity
5) Outdated Information

System Issues:
1) Navigation Problems
2) Technical Access Failures
3) Data Mismatch
4) System Latency or Failure

Communication Issues:
1) Unclear Communication
2) Lack of Specific Guidance
3) Multiple Unrelated Issues
4) Misalignment with Caller’s Needs
"""

def process_topic(label, df):
    """
    Function to process one topic label.
    Returns JSON result string.
    """
    df_subset = df[df["topic_label"] == label].reset_index(drop=True)
    df_subset = df_subset[
        (df_subset["Feedback"] == 0) | (df_subset["Feedback"] == "negative")
    ]
    result = find_errors_by_subset(df_subset)
    return label, result

import math

def find_errors_in_batches(csv_link = "../Agent Assist ML Pipeline/data/processed/with_topics.csv", batch_size=10, start_batch=0, csv_store_file="llm_responses.jsonl"):
    df = pd.read_csv(csv_link)
    json_list = []

    topic_labels = df["topic_label"].unique()
    print("Total unique topics:", len(topic_labels))
    print("The unique topics:", topic_labels)

    # Calculate batches
    num_batches = math.ceil(len(topic_labels) / batch_size)
    
    print("Total batches:", num_batches)
    
    # Determine which batch to start with
    batch_start_idx = start_batch * batch_size
    batch_end_idx = batch_start_idx + batch_size
    
    # Slice the topics to process just this batch
    batch_labels = topic_labels[batch_start_idx:batch_end_idx]
    
    print("Processing batch:", start_batch)
    print("Topics in this batch:", batch_labels)

    for label in batch_labels:
        print("Currently processing the following label:", label)
        
        df_subset = df[df["topic_label"] == label].reset_index(drop=True)
        df_subset = df_subset[
            (df_subset["Feedback"] == 0) | (df_subset["Feedback"] == "negative")
        ]

        
        if df_subset.empty:
            print(f"Skipping {label}: no negative feedback rows.")
            continue
        
        try:
            # maybe put a loop to keep going until the LLM returns a meaningful value?
            
            summaries = df_subset["Knowledge_Answer"].tolist()
            print("The summaries that will be dissected: ", summaries[0])
            

            # this has now become
            # comma separated
            llm_returned_value = False
            while (not llm_returned_value):
                try:
                    possible_errors = find_errors_by_subset(df_subset)
                    #delay the call to allow for 20 calls per minute
                    time.sleep(3)
                    llm_returned_value = True
                except Exception as e:
                    time.sleep(3)

            print("LLM Printed:", possible_errors)

            list_of_errors = [i.strip() for i in possible_errors.split(",") if i.strip()]
            all_rows = []
            for error in list_of_errors:
                all_rows.append({
                    "Label" : label,
                    "Error" : error
                })
            
            with open(csv_store_file, "a", newline="", encoding="utf-8") as csvfile:
                fieldnames = ["Label", "Error", "Parent Label"]
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                file_empty = not os.path.exists(csv_store_file) or os.stat(csv_store_file).st_size == 0
                if file_empty:
                    writer.writeheader()
                writer.writerows(all_rows)
        except Exception as e:
            print("❌ Error processing", label, ":", e)

    print("Finished Fully!")
        
        


def find_errors_parallel(csv_link="../Agent Assist ML Pipeline/data/processed/with_topics.csv", max_workers=4):
    df = pd.read_csv(csv_link)
    json_list = []
    topic_labels = df["topic_label"].unique()
    print("Topic labels:", topic_labels)

    # Use ThreadPoolExecutor (or ProcessPoolExecutor if your workload is CPU-bound)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        futures = {
            executor.submit(process_topic, label, df): label
            for label in topic_labels
        }
        counter = 0
        for future in as_completed(futures):
            label = futures[future]
            try:
                print("="*80)
                print("LABEL:", label)
                print("Filtered subset shape:", df_subset.shape)
                print("Subset preview:")
                print(df_subset.head(5))
                print("="*80)
                label, possible_errors = future.result()
                print(f"✅ Completed label: {label}")
                print("LLM Printed:", possible_errors)
                json_list.append(possible_errors)

                # Incremental write to file
                with open("llm_responses.jsonl", "a", encoding="utf-8") as f:
                    f.write(json.dumps(possible_errors))
                    f.write("\n")
            except Exception as e:
                print(f"❌ Error processing {label}: {e}")
            counter += 1
            print(f"{len(topic_labels) - counter} labels remaining.")

    # After all complete
    print("Exporting to CSVs...")

    export_all_taxonomies_to_csv(json_list)

def find_errors(csv_link = "../Agent Assist ML Pipeline\data\processed\with_topics.csv"):
    df = pd.read_csv(csv_link)
    json_list = []
    
    #get our unique topic labels
    topic_labels = df["topic_label"].unique()
    print(topic_labels)
    counter = 0
    for label in topic_labels:
        counter += 1
        print("On label: " , {label} , "with " , {len(topic_labels) - counter}, " left")
        df_subset = df[df["topic_label"] == label].reset_index(drop=True)
        # Filter only negative calls
        df_subset = df_subset[
            (df["Feedback"] == 0) | (df["Feedback"] == "negative")
        ]
        possible_errors = find_errors_by_subset(df_subset)
        print("LLM Printed: ", possible_errors)
        json_list.append(possible_errors)

        # open file in append mode
        with open("llm_responses.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(possible_errors))
            f.write("\n")  # newline

    # This is how we store our column of topics
    df["topic_label"]
    print("exporting to csvs")

    export_all_taxonomies_to_csv(json_list)


    
def develop_error_taxonomy(
    csv_link="../Agent Assist ML Pipeline/data/processed/cleaned_feedback.csv",
    max_tokens=131071
):
    import pandas as pd

    # Load data
    df = pd.read_csv(csv_link)

    # Filter negative calls
    df_subset = df[
        (df["Feedback"] == 0) | (df["Feedback"] == "negative")
    ]

    # Sample rows safely
    sample_n = min(300, len(df_subset))
    sample_df = df_subset.sample(n=sample_n, random_state=42)

    # Batching
    batch_size = 20
    batches = [
        sample_df.iloc[i:i+batch_size]
        for i in range(0, len(sample_df), batch_size)
    ]

    # Instruction text
    instruction_text = (
        "Below are examples of failed support interactions.\n"
        "Please:\n"
        "1. Review all examples carefully.\n"
        "2. Identify up to 5 common reasons why these calls were unsatisfactory.\n"
        "3. For each reason, provide:\n"
        "- a short label\n"
        "- a clear definition\n"
        "- 1–2 example excerpts illustrating it\n"
        "Return JSON with:\n"
        "categories (list of objects with label, definition, examples)."
    )

    system_text = (
        "You are a critical thinking assistant who analyzes sets of support call summaries "
        "and identifies patterns, specifically the root causes as to why the call failed, and recommends improvements. "
        "Be specific and avoid generalities."
    )

    # Process each batch
    for batch in batches:
        numbered_summaries = "\n".join(
            [f"{j+1}) '{text}'" for j, text in enumerate(batch["Knowledge_Answer"])]
        )

        prompts = [
            instruction_text,
            f"Here are {len(batch)} examples:\n\n{numbered_summaries}"
        ]

        result = prompt_llm(prompts=prompts, system_prompt=system_text)
        print(result)


def find_errors_by_subset(df_subset, max_chars = 131071):
    """
    For a subset DataFrame, aggregate knowledge_answers,
    build a prompt, and ask the LLM to analyze errors.
    """
    topic_name = df_subset["topic_label"]
    # Build a list of the text summaries, numbered
    summaries = df_subset["Knowledge_Answer"].tolist()
    
    # Format the summaries as numbered list
    numbered_text = "\n".join(
        [f"{i+1}) '{text}'" for i, text in enumerate(summaries)]
    )
    

    system_text= (
        "You are a critical thinking assistant who analyzes sets of support call summaries "
        "and identifies patterns, specifically the root causes as to why the call failed, and recommends improvements. "
        "Be specific and avoid generalities."
    )
    
         # This is the analysis instruction
    instruction_text = (
        f"""
        You are an expert error taxonomy analyst.

        Below is a list of  conversation summaries for the topic: {topic_name}.

        Please:
        1. Identify the most common failure categories from this error taxonomy:
        {derived_error_taxonomy}""" +
        """
        2. For each failure category:
        - Estimate the percentage of examples belonging to this category.
        - Provide 2–3 representative examples as excerpts.
        - Indicate whether a knowledge article was provided to the caller.
        - If an article was provided, explain why it did not resolve the issue.
        3. If any examples do not fit existing categories, propose a new category with label and definition.
        4. Return your analysis in JSON format:

        {
        "topic": "...",
        "error_categories": [
            {
            "label": "...",
            "percentage": ...,
            "examples": [
                {
                "excerpt": "...",
                "article_provided": true,
                "article_failure_reason": "..."
                },
                ...
            ]
            },
            ...
        ],
        "new_categories": [ ... ]
        }

        5. Please paraphrase excerpts for clarity.
        6. Ensure percentages total 100%.
        7. If no article is mentioned, confirm with article_provided: false. If unclear, say unknown.
        8. Avoid repeating identical excerpts across categories.
        9. Deduplicate where needed to derive more distinct categories
        10. End the JSON with a "suggestions" : which covers in specific what the top 5 subtopic issues were
        """
    )

    instruction_text = "Based on the following conversation summaries, identify all distinct key issues or failure points you observe. Return them as a comma-separated list of concise statements (max 15 words each). Be specific. Avoid generalities. Make sure each statement is different than one another. If there are no summaries to process at all then return a 0. Make sure to only use commas when separating the key issues."

    # Reserve space for instructions and buffer
    reserved_chars = len(instruction_text) + len(system_text) + 100


    # Separate rows with and without a KB article reference
    with_kb = []
    without_kb = []

    for i, row in df_subset.iterrows():
        kb_ref = row.get("Knowledge", "")
        if kb_ref and kb_ref not in ["-", "None", None, ""]:
            with_kb.append(row)
        else:
            without_kb.append(row)

    # Shuffle each list for variety

    random.shuffle(with_kb)
    random.shuffle(without_kb)

    # Combine, KB references first
    ordered_rows = with_kb + without_kb

    # Accumulate selected summaries until limit
    selected_summaries = []
    total_chars = reserved_chars
    N = 0
    for i, row in enumerate(ordered_rows):
        kb_ref = row.get("Knowledge", "")
        text = row["Knowledge_Answer"]

        # Add prefix if there was a KB article
        kb_note = ""
        if kb_ref and kb_ref not in ["-", "None", None, ""]:
            kb_note = f"(KB Article provided in transcript: {kb_ref}) \n"

        # Build the text
        new_text = f"{len(selected_summaries)+1}) '{kb_note}{text}'\n"

        # Stop if max_chars would be exceeded
        if total_chars + len(new_text) > max_chars:
            break

        selected_summaries.append(new_text)
        total_chars += len(new_text)

    # Update N to reflect how many you included
    N = len(selected_summaries)

    if not selected_summaries:
        raise ValueError("No summaries could be added without exceeding max_chars.")

    # Build numbered text
    numbered_text = "".join(selected_summaries)

   

    # We can pass multiple prompts for clarity/context
    prompts = [
        f"Here are {len(summaries)} conversation summaries:\n\n{numbered_text}",
        instruction_text
    ]
    
    # Call your prompt_llm helper (assuming you defined it as before)
    result = prompt_llm(prompts=prompts,system_prompt=system_text)
    
    return result
