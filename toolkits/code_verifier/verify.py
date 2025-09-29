from typing import List
from fuzzywuzzy import fuzz


def fim_verify_call(
    responses: List[str],
    references: List[str],
) -> List:
    assert len(responses) == len(references), (
        len(responses),
        len(references),
    )

    rewards = []
    for resp, ref in zip(responses, references):
        fuzzy_sim = fuzz.ratio(resp.strip(), ref.strip()) / 100
        rewards.append(fuzzy_sim)
    return rewards
