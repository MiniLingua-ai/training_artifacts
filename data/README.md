# Data Preparation

This folder contains scripts, notes, and configurations for preparing multilingual training data for the **MiniLingua 1B** model.  
The dataset includes 13 European languages plus programming code corpora, with extensive filtering and cleaning to ensure quality.

Data preparation relied on the [**Datatrove library (v0.4.0)**](https://github.com/huggingface/datatrove/tree/main), along with several custom scripts and processors. These handled both content filtering (e.g., removing inappropriate text) and cluster-specific preprocessing.


## Folder Contents
- `bad_words/` — multilingual lists of obscene/sexual terms used for filtering  
- `data_pipeline/` — example base pipeline for cleaning, plus custom filters
- `dataset_processing/` - individual scripts for dataset download and processing

## High-Quality Datasets

| Dataset | Links |
|---------|-------|
| EMEA-V3 | [Link](https://huggingface.co/datasets/qanastek/EMEA-V3) |
| Europarl | [Link](https://huggingface.co/datasets/Helsinki-NLP/europarl) |
| Opus Books | [Link](https://huggingface.co/datasets/Helsinki-NLP/opus_books) |
| Eurovoc | [Link](https://huggingface.co/datasets/EuropeanParliament/Eurovoc) |
| News Commentary | [Link](https://huggingface.co/datasets/Helsinki-NLP/news_commentary) |
| Eac_tm | [Link](https://huggingface.co/datasets/community-datasets/europa_eac_tm) |
| Academic Texts CLARIN | [Link](https://www.clarin.eu/resource-families/corpora-academic-texts) |
| Opus_100 | [Link](https://huggingface.co/datasets/Helsinki-NLP/opus-100) |
| Chitanka (HF) | [Link](https://huggingface.co/datasets/petkopetkov/chitanka) |
| MOSEL | [Link](https://huggingface.co/datasets/FBK-MT/mosel?row=48) |
| Wikipedia | [Link](https://huggingface.co/datasets/wikimedia/wikipedia) |
| News | [Link](https://huggingface.co/datasets/intfloat/multilingual_cc_news) |
| BNC (1951–2021) | [Link](https://dcl.bas.bg/bulnc/dostap/izteglyane/) |
| CS Academic Abstracts | [Link](https://lindat.mff.cuni.cz/repository/xmlui/handle/11234/1-1731?show=full) |
| Academic Texts | [Link](https://spraakbanken.gu.se/resurser/sweacsam) |
| Literaturbanken | [Link](https://spraakbanken.gu.se/resurser/lb-open) |
| Poems | [Link](https://spraakbanken.gu.se/resurser/poeter) |
| SVT News 2022 | [Link](https://spraakbanken.gu.se/resurser/svt-2022) |
| CLEAR Drugs and Notes | [Link](http://natalia.grabar.free.fr/resources.php) |
| Multilingual Medical Corpus | [Link](https://huggingface.co/datasets/HiTZ/Multilingual-Medical-Corpus) |
| OpenSubtitles | [Link](https://opus.nlpl.eu/OpenSubtitles/en&bg/v2018/OpenSubtitles) |
| Academic Texts 2 | [Link](https://spraakbanken.gu.se/resurser/sweachum) |
| Swedish Books | [Link](https://spraakbanken.gu.se/resurser/runeberg-biblblad) |
| WMT17 | [Link](https://huggingface.co/datasets/wmt/wmt17) |
| Yle News | [Link](https://www.kielipankki.fi/download/YLE/) |
| CurliCat Polish Corpus (ELRC-SHARE) | [Link](https://elrc-share.eu/repository/browse/curlicat-polish-corpus/f63ae912553911ed9c1a00155d02670648c0a234e0314895b52169af2af57dd7/) |
| Clarin-PL Corpus | [Link](https://clarin-pl.eu/dspace/handle/11321/699) |
| European Language Grid Corpus | [Link](https://live.european-language-grid.eu/catalogue/corpus/1279) |
| Czech SYN2015 | [Link](https://lindat.mff.cuni.cz/repository/items/2d4b6f01-c80c-4a7e-956f-642a8fa42b74) |
| ParlSpeechv2 | [Link](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/L4OAKN) |
| Ylilauta (Kielipankki) | [Link](https://www.kielipankki.fi/corpora/ylilauta/) |
| JRC-Acquis | [Link](https://clarino.uib.no/comedi/editor/lb-2017020801) |
| Suomi24 (2018-2020) | [Link](https://clarino.uib.no/comedi/editor/lb-2021101521) |
| BG News | [Link](https://dcl.bas.bg/BulNC-registration/#feeds/page/3) |
| OpenCulture | [Link](https://huggingface.co/collections/PleIAs/openculture-65d46e3ea3980fdcd66a5613) |

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
   - Translations of Datatrove's bad words list (Penedo et al., 2024) via Google Translate  
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


