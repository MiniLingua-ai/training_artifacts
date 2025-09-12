import re
import os

from numpy.random import default_rng

from datatrove.data import Document
from datatrove.pipeline.filters.base_filter import BaseFilter
from datatrove.pipeline.writers.disk_base import DiskWriter


_BADWORDS_LANGS = ['bg', 'cs', 'de', 'el', 'en', 'es', 'fi', 'fr', 'it', 'nl', 'pl', 'pt', 'sv']
PATH = './bad_words'


class CustomBadWordsFilter(BaseFilter):
    """
    Badwords filter from C4.
    Args:
        keep_fraction (float): what percentage of pages containing bad words should be kept
        fail_on_missing_language (bool) whether to fail when a document has an unknown language
        seed (int): used for the uniform distribution generator for use with keep_fraction
        default_language (str): what language for samples without language in their metadata
    """

    name = "⛰ CustomBadWordsFilter"

    def __init__(
        self,
        keep_fraction: float = 0.0,
        fail_on_missing_language: bool = True,
        seed: int = None,
        default_language: str = "en",
        exclusion_writer: DiskWriter = None,
    ):
        super().__init__(exclusion_writer)
        self.keep_fraction = keep_fraction
        self.fail_on_missing_language = fail_on_missing_language
        self._badwords_regex: dict[str, re.Pattern] = {}
        self.uniform = default_rng(seed).uniform
        self.default_language = default_language

    def _get_badwords(self, lang: str):
        if lang not in self._badwords_regex:
            if lang not in _BADWORDS_LANGS:
                if self.fail_on_missing_language:
                    raise ValueError(
                        f'There is not badwords list available for "{lang}". '
                        f"Set fail_on_missing_language=False to continue anyway."
                    )
                else:
                    return None
            local_path = os.path.join(PATH, f"{self.default_language}.txt")
            badwords: set[str] = set()
            # load from file
            with open(local_path, "rt") as f:
                badwords.update(line.strip() for line in f)

            words = [re.escape(w) for w in badwords]
            self._badwords_regex[lang] = (
                # For Japanese, Thai, and Chinese, do not require word separations.
                re.compile("|".join(words))
                if lang in ("ja", "th", "zh")
                # For other languages, match only when flanked by non-word chars.
                else re.compile(r"(?:\W|^)({})(?:\W|$)".format("|".join(words)))
            )
        return self._badwords_regex[lang]

    def filter(self, doc: Document) -> bool | tuple[bool, str]:
        lang: str = doc.metadata.get("language", self.default_language)
        badwords_regex = self._get_badwords(lang)
        if badwords_regex is None:
            self.stat_update("missing_badwords_lang", f"missing_badwords_lang_{lang}")
            return True
        badwords_found = badwords_regex.findall(doc.text.lower())
        if badwords_found and len(badwords_found) >= 5:
            self.stat_update("documents_with_badwords", f"documents_with_badwords_{lang}")
            if self.keep_fraction > 0.0 and self.uniform() < self.keep_fraction:
                self.stat_update("document_kept_with_badwords", f"document_kept_with_badwords_{lang}")
                return True
            self.stat_update(f"document_removed_with_badwords_{lang}")
            return False, "document_removed_with_badwords"
        return True