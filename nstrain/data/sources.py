"""The MiniCPM open corpora used for pre-training, one entry per (repository, directory).

``field`` is the text column; ``domain`` is the mixture domain (web_en / web_zh / math / code / long); ``tier`` follows
the UltraData tiers (L1 filtered, L2 model-selected, L3 rewritten / synthesised).  Sizes are those of the Hugging Face
listing on 2026-10-21; token counts come from the calibration (``nstrain data calibrate``) in the active tokenizer.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple


@dataclass(frozen=True)
class Source:
    repo: str
    prefix: str
    field: str
    domain: str
    tier: str
    note: str = ""
    fields: Tuple[str, ...] = ()        # several text columns joined with blank lines (e.g. the four-part code L3)


SOURCES: Dict[str, Source] = {
    # web, English
    "web_en_l2": Source("openbmb/Ultra-FineWeb", "data/ultrafineweb_en_v1_4", "content", "web_en", "L2",
                        "Ultra-FineWeb-en v1.4, 6.7 TB"),
    "web_en_l2_recent": Source("openbmb/Ultra-FineWeb", "data/ultrafineweb_l1_en_hq", "content", "web_en", "L2",
                               "L2 selection of Ultra-FineWeb-L1 (Common Crawl up to CC-MAIN-2025-51), 478 GB"),
    "web_en_l3_qa": Source("openbmb/Ultra-FineWeb-L3", "data/ultrafineweb_en_l3/qa", "content", "web_en", "L3",
                           "source text + Q&A pairs, 663 GB"),
    "web_en_l3_style": Source("openbmb/Ultra-FineWeb-L3", "data/ultrafineweb_en_l3/multi_style", "content", "web_en",
                              "L3", "encyclopedia / textbook / blog / summary rewrites, 595 GB"),
    "web_en_ultrax": Source("openbmb/UltraX-Preview", "data", "cleaned_content", "web_en", "L3",
                            "function-call refinement of five English corpora, 487 GB"),
    # web, Chinese
    "web_zh_l2": Source("openbmb/Ultra-FineWeb", "data/ultrafineweb_zh", "content", "web_zh", "L2",
                        "Ultra-FineWeb-zh, 324 GB"),
    "web_zh_l3_qa": Source("openbmb/Ultra-FineWeb-L3", "data/ultrafineweb_zh_l3/qa", "content", "web_zh", "L3",
                           "333 GB"),
    "web_zh_l3_style": Source("openbmb/Ultra-FineWeb-L3", "data/ultrafineweb_zh_l3/multi_style", "content", "web_zh",
                              "L3", "308 GB"),
    # math
    "math_l1": Source("openbmb/UltraData-Math", "data/UltraData-Math-L1", "content", "math", "L1", "367 GB"),
    "math_l2": Source("openbmb/UltraData-Math", "data/UltraData-Math-L2-preview", "content", "math", "L2", "63 GB"),
    "math_l3": Source("openbmb/UltraData-Math", "data/UltraData-Math-L3", "content", "math", "L3",
                      "conversation / multi-style / QA / textbook-exercise synthesis, 123 GB"),
    # code
    "code_l2": Source("openbmb/UltraData-Code", "data/UltraData-Code-L2", "content", "code", "L2",
                      "11 languages, 611 GB"),
    "code_l2_stack": Source("openbmb/UltraData-Stack-v3-L2", "data/UltraData-Stack-v3-L2", "content", "code", "L2",
                            "273 GB"),
    "code_l3": Source("openbmb/UltraData-Code", "data/UltraData-Code-L3", "full_content", "code", "L3",
                      "task / analysis / solution / test aligned (full_content = all four parts), 605 GB"),
    # long context (32K documents)
    "long_mix": Source("openbmb/InfLLM-V2-data-5B", "train", "text", "long", "L2",
                       "web / code / papers / wiki long documents, 16 GB"),
}
