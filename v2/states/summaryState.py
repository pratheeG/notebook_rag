from typing import List, TypedDict


class SummaryState(TypedDict):
    contents: List[str]      # The raw document chunks
    summaries: List[str]     # The individual "mapped" summaries
    final_summary: str       # The final consolidated result