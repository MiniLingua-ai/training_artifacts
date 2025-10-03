# Data Preparation

This folder contains scripts, notes, and configurations for preparing multilingual training data for the **MiniLingua 1B** model.  
The dataset includes 13 European languages plus programming code corpora, with extensive filtering and cleaning to ensure quality.

Data preparation relied on the [**Datatrove library (v0.4.0)**](https://github.com/huggingface/datatrove/tree/main), along with several custom scripts and processors. These handled both content filtering (e.g., removing inappropriate text) and cluster-specific preprocessing.


## Folder Contents
- `bad_words/` — multilingual lists of obscene/sexual terms used for filtering  
- `data_pipeline/` — example base pipeline for cleaning, plus custom filters


## Cleaning & Filtering Pipeline

Different datasets required different filtering rules, but the **full pipeline** included:

1. **Language identification** using FastText models (Joulin et al., 2016).  
2. **Heuristic filtering** (Rae et al., 2022), such as:
   - Removing overly short/long documents  
   - Filtering by % of bullet lines  
   - Filtering by % of non-alphanumeric characters  
3. **Repetition filter** to remove documents with excessive repetition of characters, words, or n-grams (Rae et al., 2022).  
4. **Blacklist filtering** using a multilingual list of inappropriate terms:
   - Based on [LDNOOBW](https://github.com/LDNOOBW/List-of-Dirty-Naughty-Obscene-and-Otherwise-Bad-Words)  
   - Translations of Datatrove’s bad words list (Penedo et al., 2024) via Google Translate  
   - Extended and manually reviewed with help from native speakers  
5. **Deduplication within datasets** to remove near-duplicates with high line or paragraph overlap (Rae et al., 2022).  
6. **Cross-set deduplication** using Jaccard similarity to prevent overlap between training and evaluation splits, reducing leakage and ensuring fair downstream evaluation.


## Token Estimation Prior to Tokenizer Training

To estimate the number of tokens in the dataset before training the MiniLingua tokenizer, we used language-specific BERT models as a lower boundary for token fertility.  
Since BERT tokenizers tend to produce fewer tokens per word compared to LLM tokenizers, the actual number of tokens expected with our custom tokenizer will be higher.  

---

## Models Used for Token Estimation

| Language   | Model |
|------------|------------------------------------------------------------------|
| Bulgarian  | `rmihaylov/bert-base-bg` |
| Czech      | `ufal/robeczech-base` |
| Dutch      | `GroNLP/bert-base-dutch-cased` |
| English    | `google-bert/bert-base-uncased` |
| Finnish    | `TurkuNLP/bert-base-finnish-cased-v1` |
| French     | `almanach/camembert-base` |
| German     | `google-bert/bert-base-german-cased` |
| Greek      | `dimitriz/greek-media-bert-base-uncased` |
| Italian    | `nickprock/sentence-bert-base-italian-uncased` |
| Polish     | `dkleczek/bert-base-polish-cased-v1` |
| Portuguese | `neuralmind/bert-base-portuguese-cased` |
| Spanish    | `dccuchile/bert-base-spanish-wwm-cased` |
| Swedish    | `AI-Nordics/bert-large-swedish-cased` |
| Code  | `Qwen/Qwen2.5-Coder-7B-Instruct` |

---

## High-Quality Data Token Counts

| Dataset          | Bulgarian      | Czech          | Dutch          | English        | Finnish        | French         | German         | Greek          | Italian        | Polish         | Portuguese     | Spanish        | Swedish        |
|------------------|----------------|----------------|----------------|----------------|----------------|----------------|----------------|----------------|----------------|----------------|----------------|----------------|----------------|
| **Before cleaning** | 3,995,673,031  | 8,536,684,340  | 3,861,865,573  | 80,674,561,726 | 3,265,195,144  | 26,127,824,156 | 110,944,202,001 | 3,057,476,347  | 15,477,385,342 | 9,689,963,261  | 7,552,438,575  | 25,597,712,237 | 1,873,810,365  |
| **After cleaning** | 1,981,601,932  | 5,521,693,018  | 3,288,768,869  | 64,137,205,406 | 1,825,365,557  | 15,143,503,017 | 24,828,122,458 | 2,352,751,838  | 11,016,431,773 | 7,230,903,713  | 6,213,874,420  | 23,645,292,869 | 1,567,406,870  |

---

## Web Data Token Counts

| Dataset          | Bulgarian      | Czech          | Dutch          | English        | Finnish        | French         | German         | Greek          | Italian        | Polish         | Portuguese     | Spanish        | Swedish        |
|------------------|----------------|----------------|----------------|----------------|----------------|----------------|----------------|----------------|----------------|----------------|----------------|----------------|----------------|
| **Before cleaning** | 11,406,119,433 | 30,363,362,800 | 62,592,537,766 | 325,827,385,301 | 20,022,548,797 | 177,706,269,060 | 213,561,976,993 | 12,573,191,127 | 78,645,484,032 | 50,557,315,232 | 107,275,196,495 | 177,460,679,884 | 22,921,582,149 |
| **After cleaning**  | 10,457,785,665 | 25,739,638,912 | 49,000,070,823 | 305,132,707,615 | 13,974,542,750 | 148,916,819,674 | 201,869,821,732 | 11,699,045,700 | 74,853,451,776 | 47,607,730,311 | 85,005,821,112  | 155,833,549,189 | 16,809,318,858 |

---


