def drop_columns_with_many_nans(df, threshold=0.2):
    """
    删除缺失值比例超过指定阈值的列。

    参数：
    - df：pandas DataFrame，需要处理的数据。
    - threshold：浮点数，表示缺失值比例的阈值，默认为 0.2（即 20%）。

    返回：
    - 处理后的 DataFrame，已删除指定列。
    """
    nan_ratio = df.isna().mean()
    cols_to_drop = nan_ratio[nan_ratio > threshold].index
    df_cleaned = df.drop(columns=cols_to_drop)

    return df_cleaned
