import argparse
import json
from pathlib import Path

LABELS = {
    'person': 'PERSON', 'person name': 'PERSON', 'PERSON': 'PERSON',
    'organization': 'ORGANIZATION', 'organization name': 'ORGANIZATION',
    'ORGANIZATION': 'ORGANIZATION',
    'location': 'LOCATION', 'geographical location': 'LOCATION',
    'LOCATION': 'LOCATION',
}


# Validate offsets and create an exact-span comparison key.
def mention_key(mention: dict, case: dict) -> tuple:
    start, end = mention['start'], mention['end']
    if not isinstance(start, int) or not isinstance(end, int):
        raise ValueError('Offsets must be integers.')
    if not 0 <= start < end <= len(case['text']):
        raise ValueError(f"Invalid offsets in {case['item_ref']}")
    if case['text'][start:end] != mention['text']:
        raise ValueError(f"Source text mismatch in {case['item_ref']}")
    label = LABELS[mention['label']]
    return case['source_file'], case['item_ref'], start, end, label


# Score only reviewed passage IDs; ignore predictions outside that subset.
def score(gold: dict, predictions: list[dict]) -> dict:
    cases = {(c['source_file'], c['item_ref']): c for c in gold['cases']}
    if len(cases) != len(gold['cases']):
        raise ValueError('Duplicate gold passage IDs.')
    expected = {mention_key(e, c) for c in cases.values() for e in c['entities']}
    actual = set()
    ignored = 0
    for prediction in predictions:
        case = cases.get((prediction['source_file'], prediction['item_ref']))
        if case is None:
            ignored += 1
            continue
        actual.add(mention_key(prediction, case))
    tp = len(expected & actual)
    fp = len(actual - expected)
    fn = len(expected - actual)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {
        'true_positives': tp, 'false_positives': fp, 'false_negatives': fn,
        'precision': round(precision, 4), 'recall': round(recall, 4),
        'f1': round(2 * precision * recall / (precision + recall), 4)
        if precision + recall else 0.0,
        'ignored_predictions_outside_subset': ignored,
        'false_positive_spans': sorted(actual - expected),
        'missed_gold_spans': sorted(expected - actual),
    }


# Compare one or more existing prediction files against the draft gold set.
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('gold', type=Path)
    parser.add_argument('predictions', type=Path, nargs='+')
    args = parser.parse_args()
    gold = json.loads(args.gold.read_text(encoding='utf-8'))
    if gold.get('status') == 'draft_requires_human_review':
        print('DRAFT ANNOTATIONS: review before treating metrics as authoritative.')
    for path in args.predictions:
        predictions = json.loads(path.read_text(encoding='utf-8'))
        result = score(gold, predictions)
        print(f'\n{path.name}')
        print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
