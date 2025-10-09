import os
import glob
import pandas as pd
from datatrove.data import Document
from datatrove.pipeline.filters.base_filter import BaseFilter
from datatrove.utils.typeshelper import Languages
from sklearn.feature_extraction.text import CountVectorizer
import re


class TestDataFilter(BaseFilter):
    name = "📄 Test Data Filter"

    def __init__(self, test_data_path: str, threshold: float = 0.8):
        """
        Filters out documents that have a high Jaccard similarity with test documents.
        
        Args:
            test_data_path: Path to the folder containing test documents in .parquet format.
            language: Language used for tokenization.
            threshold: Jaccard similarity threshold for filtering (default: 0.8).
        """
        super().__init__()
        self.threshold = threshold
        self.vectorizer = CountVectorizer(analyzer='word', ngram_range=(13, 13), binary=True)
        self.test_ngram_matrix = self._load_test_data(test_data_path)
        self.n_test_ngrams = self.test_ngram_matrix.sum(axis=1)
    
    @staticmethod
    def _find_words(text):
        pattern = r"(?u)\b\w\w+\b"  # Matches words of at least two characters
        return re.findall(pattern, text)

    def _load_test_data(self, path: str):
        """
        Reads all .parquet test documents and extracts their 13-grams using CountVectorizer.
        """
        test_texts = []
        for file in glob.glob(os.path.join(path, "*.parquet")):
            df = pd.read_parquet(file)
            if "text" in df.columns:
                test_texts.extend(df["text"].tolist())
        
        return self.vectorizer.fit_transform(test_texts)

    def filter(self, doc: Document) -> bool | tuple[bool, str]:
        """
        Filters documents based on Jaccard similarity with test data.
        """
        doc_ngram_vector = self.vectorizer.transform([doc.text])
        n_train_ngrams = max(1, len(self._find_words(doc.text)) - 12)
        
        if doc_ngram_vector.sum() == 0:
            return True  # No valid n-grams, accept the document

        # Compute Jaccard similarity 
        intersection = (doc_ngram_vector.multiply(self.test_ngram_matrix)).sum(axis=1)
        # Let's think that 13-grams are unique
        union = self.n_test_ngrams + n_train_ngrams - intersection
        similarity_scores = intersection / union
        max_similarity = float(similarity_scores.max())
        
        if max_similarity > self.threshold:
            return False, "test_data_similarity"
        
        return True
