import pandas as pd


def _as_tuple_group_key(group_key) -> tuple:
    if isinstance(group_key, tuple):
        return group_key

    return (group_key,)


def compute_group_targets(
    df: pd.DataFrame,
    sample_size: int,
    group_columns: list[str],
) -> dict[tuple, int]:
    if not group_columns:
        return {("__all__",): min(sample_size, len(df))}

    if group_columns[0] == "world_region" and len(group_columns) > 1:
        primary_targets = compute_group_targets(df, sample_size, ["world_region"])
        targets = {}

        for primary_key, primary_target in primary_targets.items():
            primary_value = primary_key[0]
            primary_df = df[df["world_region"] == primary_value]
            secondary_targets = compute_group_targets(
                primary_df,
                primary_target,
                group_columns[1:],
            )

            for secondary_key, secondary_target in secondary_targets.items():
                targets[primary_key + secondary_key] = secondary_target

        return targets

    groups = list(df.groupby(group_columns, sort=True, dropna=False))
    group_sizes = {
        _as_tuple_group_key(group_key): len(group)
        for group_key, group in groups
    }
    targets = dict.fromkeys(group_sizes, 0)

    while sum(targets.values()) < min(sample_size, len(df)):
        grew = False
        for group_key in sorted(group_sizes):
            if sum(targets.values()) >= min(sample_size, len(df)):
                break

            if targets[group_key] >= group_sizes[group_key]:
                continue

            targets[group_key] += 1
            grew = True

        if not grew:
            break

    return targets
