from pathlib import Path
import pandas as pd
import polars as pl
import xmltodict


def read_semeval_xml(filepath: Path) -> pd.DataFrame:
    with open(filepath, "rb") as f:
        parsed_dict = xmltodict.parse(f.read())

    sentences_list = parsed_dict["sentences"]["sentence"]

    for sentence in sentences_list:
        sentence["sentence_id"] = sentence["@id"]
        sentence["aspect_terms"] = []

        if "aspectTerms" in sentence:
            aspect_terms = sentence["aspectTerms"]["aspectTerm"]
            if isinstance(aspect_terms, dict):
                aspect_terms = [aspect_terms]
            for aspect_term in aspect_terms:
                aterm = {
                    "to": aspect_term["@to"],
                    "from": aspect_term["@from"],
                    "term": aspect_term["@term"],
                    "polarity": aspect_term["@polarity"],
                }
                sentence["aspect_terms"].append(aterm)

    df = pd.DataFrame(sentences_list)[["sentence_id", "text", "aspect_terms"]]

    return df


files = [
    {
        "domain": "laptops",
        "split": "train",
        "path": "data/semeval_2014/laptops_train.xml",
    },
    {
        "domain": "laptops",
        "split": "test",
        "path": "data/semeval_2014/laptops_test.xml",
    },
    {
        "domain": "restaurants",
        "split": "train",
        "path": "data/semeval_2014/restaurants_train.xml",
    },
    {
        "domain": "restaurants",
        "split": "test",
        "path": "data/semeval_2014/restaurants_test.xml",
    },
]

# Read all files and merge them into one DataFrame
dfs = []

for file in files:
    df = read_semeval_xml(Path(file["path"]))
    df["domain"] = file["domain"]
    df["split"] = file["split"]
    dfs.append(df)

df_complete = pl.from_pandas(pd.concat(dfs, ignore_index=True))


# Remove examples that contain the conflict polarity
def has_conflict(aspect_terms):
    for term in aspect_terms:
        if term["polarity"] == "conflict":
            return True
    return False


df_clean = df_complete.filter(
    ~pl.col("aspect_terms").map_elements(
        lambda x: has_conflict(x), return_dtype=pl.Boolean
    )
)

df_clean.write_parquet("data/semeval_2014/cleaned.parquet")
