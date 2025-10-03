import sys

from datatrove.executor.slurm import SlurmPipelineExecutor
from datatrove.pipeline.filters import (
    GopherQualityFilter,
    GopherRepetitionFilter,
    LanguageFilter,
)
from datatrove.pipeline.tokens import TokensCounter
from datatrove.pipeline.formatters import FTFYFormatter, SymbolLinesFormatter
from datatrove.pipeline.readers.parquet import ParquetReader
from datatrove.pipeline.writers.parquet_id_writer import ParquetWriterNoText
from bad_words_filter import CustomBadWordsFilter


DUMP = 'high_quality'
MAIN_OUTPUT_PATH = "datatrove_output"
LANG = 'bg'
LANG_ISO = 'bul'
TOKEN_COUNTER = 'rmihaylov/bert-base-bg'
PARTITION = 'batch_hsw'

executor = SlurmPipelineExecutor(
    job_name=f"{DUMP}_{LANG}_news",
    pipeline=[
        ParquetReader(
            data_folder="/scratch/cs/small_lm/open_subtitles",
            glob_pattern=f"{LANG}_1_0.parquet",
            default_metadata={"dump": DUMP},
        ),
        TokensCounter(TOKEN_COUNTER),
        LanguageFilter(
            languages=LANG,
            language_threshold=0.65,
        ),
        TokensCounter(TOKEN_COUNTER),
        CustomBadWordsFilter(default_language=LANG),
        TokensCounter(TOKEN_COUNTER),
        SymbolLinesFormatter(),
        GopherRepetitionFilter(top_n_grams = ((2, 0.4), (3, 0.3), (4, 0.3)),
                                dup_n_grams = ((5, 0.3), (6, 0.3), (7, 0.2), (8, 0.2), (9, 0.2), (10, 0.2)),
                                ),
        GopherQualityFilter(min_doc_words=2,
                            language=LANG_ISO,
                            min_stop_words=None,
                            max_non_alpha_words_ratio=0.6,
                            max_bullet_lines_ratio=0.95,
                           ),
        FTFYFormatter(),
        TokensCounter(TOKEN_COUNTER),
        ParquetWriterNoText(f"{MAIN_OUTPUT_PATH}/output/{DUMP}/{LANG}/sub"),
    ],
    tasks=1,
    time="2-00:00:00",
    logging_dir=f"{MAIN_OUTPUT_PATH}/logs/{DUMP}/{LANG}/sub",
    slurm_logs_folder=f"{MAIN_OUTPUT_PATH}/slurm_logs/{DUMP}/{LANG}/sub",
    randomize_start_duration=180,
    mem_per_cpu_gb=20,
    cpus_per_task=6,
    partition=PARTITION,
)
executor.run()
