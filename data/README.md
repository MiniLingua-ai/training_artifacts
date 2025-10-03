# Data Preparation

This folder contains scripts, notes, and configurations for preparing multilingual training data for the **MiniLingua 1B** model.  
The dataset covers **13 European languages** and **code corpora**, carefully filtered and cleaned to ensure quality.  

## Contents
- **Preprocessing scripts** – text normalization, deduplication, filtering.  
- **Dataset configs** – specifying language splits and proportions.
- **Loading scripts** – data extraction and format conversion utilities.

## 📁 Dataset Processing Scripts

This folder contains various loading and preprocessing scripts for different data sources used in MiniLingua 1B training.
These scripts handle format conversion, text cleaning, language detection, and quality filtering for the diverse multilingual datasets used in training.  

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


