import pandas as pd
from tqdm import tqdm
from src.models.label_model import TaxonomyLabeler
import matplotlib.pyplot as plt
from IPython.display import display

def preprocess_conversation_times_dataframe(
    df: pd.DataFrame,
    timestamp_column: str = "Timestamp"
) -> pd.DataFrame:
    """
    Preprocesses a conversation dataframe by:
    - Converting timestamp to datetime
    - Extracting day of week and hour of day

    Parameters:
    -----------
    df : pd.DataFrame
        Your dataframe containing conversation data.
    text_column : str
        The name of the column containing conversation text.

    Returns:
    --------
    pd.DataFrame
        A copy of the dataframe with new columns:
        'day_of_week', 'hour_of_day', 'conversation_length'
    """

    

    # Make a copy so original df isn't modified
    df = df.copy()

    print(f"Parsing {len(df)} timestamps with progress bar...")

    parsed_dates = []
    for ts in tqdm(df[timestamp_column], desc="Parsing timestamps"):
        parsed = pd.to_datetime(
            ts,
            format="%b %d, %Y, %I:%M:%S %p"
        )
        parsed_dates.append(parsed)

    df[timestamp_column] = parsed_dates

    # Extract day of the week (e.g., Monday)
    df["day_of_week"] = df[timestamp_column].dt.day_name()

    # Extract hour of day (0-23)
    df["hour_of_day"] = df[timestamp_column].dt.hour

    return df

def preprocess_conversation_length_dataframe(
    df: pd.DataFrame,
    text_column: str = "Knowledge_Answer",
) -> pd.DataFrame:
    """
    Preprocesses a conversation dataframe by:
    - Computing conversation length in words

    Parameters:
    -----------
    df : pd.DataFrame
        Your dataframe containing conversation data.
    timestamp_column : str
        The name of the column containing timestamps.

    Returns:
    --------
    pd.DataFrame
        A copy of the dataframe with new columns:
        'conversation_length'
    """
    # Make a copy so original df isn't modified
    df = df.copy()
    # Compute conversation length in words
    df["conversation_length"] = df[text_column].str.split().str.len()

    return df

taxonomy_label_mapping = {
    "Errors where employees were confused about how to submit leave requests, get them approved by a manager, or understand the leave request process.": 
        "Confusion About Leave Request Submission or Approval",
    
    "Errors related to employees not understanding FMLA eligibility requirements, bonding eligibility criteria, or how to qualify for protected leave benefits.":
        "Unclear FMLA or Bonding Eligibility Criteria",
    
    "Errors where employees reported that knowledge base articles were missing, outdated, incomplete, or did not address their questions effectively.":
        "Knowledge Base Articles Missing or Not Relevant",
    
    "Errors involving paycheck discrepancies, incorrect deductions, overpayment issues, or disputes about payment amounts.":
        "Paycheck Errors, Deductions, or Overpayment Disputes",
    
    "Errors caused by employees being unable to access HR systems, forms, portals, or tools required to manage their requests.":
        "Inability to Access HR Systems or Forms",
    
    "Errors where employees were unaware of different leave types, how benefits interacted, or which type of leave applied to their situation.":
        "Employees Unaware of Leave Types or Benefit Interactions",
    
    "Errors related to difficulties providing required documentation, uploading forms, or verifying their identity for leave or benefits.":
        "Difficulty Providing Documentation or Verifying Identity",
    
    "Errors where rules for leave accrual, usage policies, or balances were not clear or consistently explained to employees.":
        "Lack of Clear Rules for Leave Accrual and Usage",
    
    "Errors involving complex or confusing enrollment processes for benefits, dependent coverage, or qualifying events.":
        "Complex or Confusing Enrollment Processes",
    
    "Errors related to unclear procedures for applying for disability insurance, understanding disability benefits, or reapplying after denials.":
        "Unclear Disability Insurance Procedures",
    
    "Errors due to delays in processing leave requests, approvals, payments, or other time-sensitive actions by HR teams.":
        "Delays in Processing or Approving Requests",
    
    "Errors stemming from inconsistent or unclear policies that vary by region, state, or business unit and cause employee confusion.":
        "Region-Specific Policy or Escalation Confusion",
    
    "Errors where communication from HR was inadequate, vague, or missing critical details employees needed to resolve their issues.":
        "Inadequate or Vague Communication to Employees",
    
    "Errors that could not be classified into any of the other categories due to insufficient detail or highly unusual circumstances.":
        "Generic or Unclassifiable Issues"
}

def label_errors_with_taxonomy(df, new_csv="final_labeled_errors.csv"):
    taxonomy_labels = [
        "Confusion About Leave Request Submission or Approval",
        "Unclear FMLA or Bonding Eligibility Criteria",
        "Knowledge Base Articles Missing or Not Relevant",
        "Paycheck Errors, Deductions, or Overpayment Disputes",
        "Inability to Access HR Systems or Forms",
        "Employees Unaware of Leave Types or Benefit Interactions",
        "Difficulty Providing Documentation or Verifying Identity",
        "Lack of Clear Rules for Leave Accrual and Usage",
        "Complex or Confusing Enrollment Processes",
        "Unclear Disability Insurance Procedures",
        "Delays in Processing or Approving Requests",
        "Region-Specific Policy or Escalation Confusion",
        "Inadequate or Vague Communication to Employees",
        "Generic or Unclassifiable Issues"
    ]
    labeler = TaxonomyLabeler(taxonomy_labels)
    error_texts = df["Knowledge_Answer"].fillna("").astype(str).tolist()
    labels, scores = labeler.label_errors(error_texts, threshold=0.3)
    df["Parent Error Topic"] = labels
    df["Parent Error Similarity Score"] = scores
    df.to_csv(new_csv, index=False)
    print("Saved to CSV")
    
    # Count rows below threshold
    threshold = 0.3
    num_below_threshold = (df["Similarity_Score"] < threshold).sum()
    num_total = len(df)
    percent_below = (num_below_threshold / num_total) * 100

      # Count rows above threshold
    num_above_threshold = num_total - num_below_threshold
    percent_above = 100 - percent_below

    # Print counts and percentages
    print("\n=== Similarity Score Summary ===")
    print(f"Total records: {num_total}")
    print(f"Records below threshold ({threshold}): {num_below_threshold} ({percent_below:.2f}%)")
    print(f"Records above threshold: {num_above_threshold} ({percent_above:.2f}%)")

    # Create a simple ASCII bar chart
    print("\n[ASCII Bar Chart]")
    bar_length = 50

    # Calculate proportional bar lengths
    below_bar = int((num_below_threshold / num_total) * bar_length)
    above_bar = bar_length - below_bar

    print(f"Below Threshold  : {'#' * below_bar}{' ' * above_bar} ({percent_below:.2f}%)")
    print(f"Above Threshold  : {'#' * above_bar}{' ' * below_bar} ({percent_above:.2f}%)")

    # Plot histogram
    plt.figure(figsize=(8, 6))
    plt.hist(df["Similarity_Score"], bins=20, color="steelblue", edgecolor="black")

    # Add threshold line
    threshold = 0.3
    plt.axvline(threshold, color="red", linestyle="--", linewidth=1.5, label=f"Threshold = {threshold}")

    # Add titles and labels
    plt.title("Distribution of Similarity Scores")
    plt.xlabel("Similarity Score")
    plt.ylabel("Number of Records")
    plt.legend()

    # Show the plot
    plt.tight_layout()
    plt.show()

def label_categories_with_taxonomy(df, new_csv="final_labeled_categories.csv"):
    taxonomy_labels = [
        'Payroll / Compensation', 
        'Leave Management / FMLA',
        'Enrollment & Benefits', 
        'Access & Technical Issues', 
        'Retirement',
        'Other / Miscellaneous',
        'HR General / Operations',
        'Taxes & Withholding',
        'Disability & State Claims', 
        'Verification & Documentation',
        'Timekeeping & Scheduling', 
        'Job Changes & Terminations']

    labeler = TaxonomyLabeler(taxonomy_labels)
    summary_texts = df["Knowledge_Answer"].fillna("").astype(str).tolist()
    labels, scores = labeler.label_errors(summary_texts, threshold=0.25)
    df["Parent Category Topic"] = labels
    df["Parent Category Similarity_Score"] = scores

    df.to_csv(new_csv, index=False)
    print("Saved to CSV")


    # Count rows below threshold
    threshold = 0.25
    num_below_threshold = (df["Similarity_Score"] < threshold).sum()
    num_total = len(df)
    percent_below = (num_below_threshold / num_total) * 100

    # Count rows above threshold
    num_above_threshold = num_total - num_below_threshold
    percent_above = 100 - percent_below

    # Print counts and percentages
    print("\n=== Similarity Score Summary ===")
    print(f"Total records: {num_total}")
    print(f"Records below threshold ({threshold}): {num_below_threshold} ({percent_below:.2f}%)")
    print(f"Records above threshold: {num_above_threshold} ({percent_above:.2f}%)")

    # Create a simple ASCII bar chart
    print("\n[ASCII Bar Chart]")
    bar_length = 50

    # Calculate proportional bar lengths
    below_bar = int((num_below_threshold / num_total) * bar_length)
    above_bar = bar_length - below_bar

    print(f"Below Threshold  : {'#' * below_bar}{' ' * above_bar} ({percent_below:.2f}%)")
    print(f"Above Threshold  : {'#' * above_bar}{' ' * below_bar} ({percent_above:.2f}%)")

    # Plot histogram
    plt.figure(figsize=(8, 6))
    plt.hist(df["Similarity_Score"], bins=20, color="steelblue", edgecolor="black")

    # Add threshold line
    threshold = 0.25
    plt.axvline(threshold, color="red", linestyle="--", linewidth=1.5, label=f"Threshold = {threshold}")

    # Add titles and labels
    plt.title("Distribution of Similarity Scores")
    plt.xlabel("Similarity Score")
    plt.ylabel("Number of Records")
    plt.legend()

    # Show the plot
    plt.tight_layout()
    plt.show()