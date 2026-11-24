from common.wine import WineSample


def process_data(data: list[WineSample]) -> list[dict]:
    """Clean incoming samples and return them as JSON-ready dicts.

    Field-level checks (types, non-negative values, valid class names) are
    already done by Pydantic. Here we drop exact duplicate samples so that a
    repeated submission doesn't get extra weight in training.
    """
    seen = set()
    processed = []
    for sample in data:
        record = sample.model_dump()
        key = tuple(record.values())
        if key in seen:
            continue
        seen.add(key)
        processed.append(record)
    return processed
