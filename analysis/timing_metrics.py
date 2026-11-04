"""Wall-clock interval operations; overlapping intervals count only once."""


def merge_intervals(intervals):
    merged = []
    for start, end in sorted(intervals):
        if end <= start:
            continue
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def overlap_duration(left, right):
    return sum(end - start for start, end in intersection_intervals(left, right))


def intersection_intervals(left, right):
    left, right = merge_intervals(left), merge_intervals(right)
    i = j = 0
    intervals = []
    while i < len(left) and j < len(right):
        start, end = max(left[i][0], right[j][0]), min(left[i][1], right[j][1])
        if end > start:
            intervals.append((start, end))
        if left[i][1] <= right[j][1]:
            i += 1
        else:
            j += 1
    return intervals
