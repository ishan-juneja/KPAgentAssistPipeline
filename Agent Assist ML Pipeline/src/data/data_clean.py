import re
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer 
import spacy

# Load spaCy model once
nlp = spacy.load("en_core_web_sm")

# Extended stopword list
CUSTOM_STOPWORDS = {
    "call", "inquire", "confirm", "advise", "explain", "provide", "contact",
    "say", "ask", "state", "tell", "speak", "discuss", "talk", "mention",
    "check", "inform", "review", "note", "receive", "submit", "process"
}

NOISE_PHRASES = [
    "the caller",
    "the agent",
    "https kphr",
    "my salesforce",
    "my salesforce com",
    "salesforce com",
    "https kphr my",
    "https kphr my salesforce",
    "kphr my",
    "kphr my salesforce",
    "kphr my salesforce com",
    "refer to",
    "refer to the",
    "for more",
    "you can",
    "com articles",
    "my salesforce com articles",
    "salesforce com articles",
    "more detailed",
    "for more detailed",
    "can refer",
    "can refer to",
    "can refer to the",
    "articles en_us",
    "articles en_us knowledge",
    "com articles en_us",
    "com articles en_us knowledge",
    "en_us knowledge",
    "salesforce com articles en_us",
    "com lightning",
    "com lightning knowledge__kav",
    "lightning knowledge__kav",
    "my salesforce com lightning",
    "salesforce com lightning",
    "salesforce com lightning knowledge__kav",
    "hrconnect portal",
    "kp org",
    "hrconnect kp",
    "hrconnect kp org",
    "https hrconnect",
    "https hrconnect kp",
    "https hrconnect kp org",
    "hrconnect kp org wps",
    "kp org wps",
    "org wps",
    "org wps poc",
    "kp org wps poc",
    "poc urile",
    "poc urile wcm",
    "articles knowledge",
    "com articles knowledge",
    "articles https",
    "articles https kphr",
    "articles https kphr my",
    "leave https",
    "leave https kphr",
    "leave https kphr my",
    "portal https",
    "hrconnect portal https",
    "hrconnect portal for",
    "portal for",
    "all regions https",
    "all regions https kphr",
    "for all regions https",
    "regions https",
    "regions https kphr",
    "regions https kphr my",
    "navigate to",
    "opt out",
    "case number",
    "here are",
    "here are the",
    "here are some",
    "the following",
    "the following resources",
    "the following steps",
    "refer to the following",
    "to the following",
    "to the following resources",
    "the hrconnect",
    "the hr connect",
    "the request",
    "the return",
    "the issue",
    "the first",
    "the first day",
    "the key",
    "the key points",
    "the next",
    "the new",
    "the required",
    "the same",
    "the appropriate",
    "the process for",
    "the necessary",
    "the hr",
    "the call",
    "the benefits",
    "the pay",
    "the form",
    "the manager",
    "agent confirmed",
    "agent confirmed that",
    "the agent confirmed",
    "the agent confirmed that",
    "agent explained",
    "agent explained the",
    "agent explained that",
    "the agent explained",
    "the agent explained the",
    "the agent explained that",
    "agent provided",
    "the agent provided",
    "agent verified",
    "the agent verified",
    "agent also",
    "the agent also",
    "caller inquired",
    "the caller inquired",
    "caller inquired about",
    "the caller inquired about",
    "inquired about",
    "inquired about the",
    "caller inquired about the",
    "follow these",
    "follow these steps",
    "follow up",
    "for further assistance",
    "further assistance",
    "to ensure",
    "to follow",
    "to confirm",
    "to check",
    "to contact",
    "to be",
    "to use",
    "to submit",
    "to work",
    "to their",
    "to the hrconnect",
    "to the hrconnect portal",
    "through the hrconnect",
    "through the hrconnect portal",
    "for assistance",
    "for this",
    "for their",
    "for all",
    "for general",
    "for more information",
    "for more details",
    "for further",
    "for fmla",
    "in general",
    "inquired about their",
    "inquired about the",
    "confirmed the",
    "confirmed that",
    "explained the",
    "explained that",
    "advised the",
    "advised the caller",
    "advised the caller to",
    "review the",
    "check the",
    "return to",
    "return from",
    "return from leave",
    "submit the",
    "must submit",
    "must submit the",
    "be submitted",
    "be taken",
    "be eligible",
    "be eligible for",
    "be required",
    "be returned",
    "be processed",
    "be provided",
    "be completed",
    "be contacted",
    "be notified",
    "be reviewed",
    "be sent",
    "be included",
    "be processed for",
    "sorry couldn",
    "sorry couldn find",
    "sorry couldn find an",
    "couldn find",
    "couldn find an",
    "couldn find an answer",
    "find an answer",
    "find an answer for",
    "an answer",
    "an answer for",
    "an answer for this",
    "answer for",
    "answer for this",
    "if the",
    "and the",
    "with the",
    "on the",
    "of the",
    "to the",
    "in the",
    "for the",
    "at the",
    "due to",
    "based on",
    "such as",
    "will be",
    "must be",
    "should be"
]

useful_keywords = [
    "fmla", "cobra", "benefit", "certification", "wage", "claim",
    "overpayment", "leave", "maternity", "disability", "1250", "1451",
    "employment", "termination", "rehire", "enrollment", "appeal"
]


noise_patterns = [
    r"^\d{2}[: ]\d{2}",              # timestamps like '10 00'
    r"amp\b",                        # HTML artifact
    r"hidenavbar",                  # UI junk
    r"wam|dmi|ui knowledge",        # internal system IDs
    r"[a-f0-9]{6,}",                # hashes/UUIDs
    r"\bhttps?://",                 # links
]

gray_keywords = [
    "calendar", "period", "rolling", "week", "month", "hour", "days",
    "documentation", "eligibility"
]

def find_common_phrases(df, column="Knowledge_Answer", ngram_range=(2, 4), min_df=20, top_n=300):
    """
    Extract and return the top N most common n-grams (by frequency) from a DataFrame column.
    """

    vectorizer = CountVectorizer(ngram_range=ngram_range, min_df=min_df)
    X = vectorizer.fit_transform(df[column])
    
    feature_names = vectorizer.get_feature_names_out()
    counts = X.sum(axis=0).A1  # Flatten the matrix to 1D array

    phrases_freq = list(zip(feature_names, counts))
    phrases_freq = sorted(phrases_freq, key=lambda x: x[1], reverse=True)

    df_phrases = pd.DataFrame(phrases_freq, columns=["phrase", "count"])

    print(df_phrases.head(50))  # Print top 20 most frequent phrases

    return df_phrases

def classify_phrase(phrase):
    phrase_l = phrase.lower()

    # Rule 1: Check if phrase matches any noise pattern
    for pattern in noise_patterns:
        if re.search(pattern, phrase_l):
            return "noise"

    # Rule 2: Check if it contains useful keywords
    for keyword in useful_keywords:
        if keyword in phrase_l:
            return "useful"

    # Rule 3: Check gray zone keywords
    for keyword in gray_keywords:
        if keyword in phrase_l:
            return "gray"

    # Default to gray if uncertain
    return "gray"

def label_ngrams(common_ngrams):
    df_phrases = pd.DataFrame(common_ngrams, columns=["phrase"])
    df_phrases["category"] = df_phrases["phrase"].apply(classify_phrase)

    # Optional: save for human review
    df_phrases.head(700).to_csv("classified_phrases.csv", index=False)

    return df_phrases


def filter_phrases(labeled_df, keep=("useful", "gray")):
    """
    Filter out phrases not matching categories in 'keep'.
    """
    labeled_df = labeled_df.copy()
    labeled_df["category"] = labeled_df["Knowledge_Answer"].apply(classify_phrase)
    return labeled_df[labeled_df["category"].isin(keep)].reset_index(drop=True)

def remove_noise_phrases(text, noise_phrases=NOISE_PHRASES):
    text_lower = text.lower()
    for phrase in noise_phrases:
        text_lower = text_lower.replace(phrase, "")
    return text_lower

def filter_phrases_by_noise(df):
    """
    Returns a copy of df with cleaned Knowledge_Answer column.
    """
    df_copy = df.copy()
    df_copy["Knowledge_Answer"] = df_copy["Knowledge_Answer"].apply(remove_noise_phrases)
    return df_copy

def extract_keywords(text):
    doc = nlp(text)
    tokens = []
    for token in doc:
        if token.pos_ in {"NOUN", "PROPN", "VERB"}:
            lemma = token.lemma_.lower()
            if lemma not in CUSTOM_STOPWORDS:
                tokens.append(lemma)
    return " ".join(tokens)

def shorten_summary(df):
    df_copy = df.copy()
    df_copy["Knowledge_Answer"] = df_copy["Knowledge_Answer"].apply(extract_keywords)
    return df_copy


