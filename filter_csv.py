import pandas as pd

def filter_csv_columns():
    """
    Filters specified columns from a CSV file and saves to two new CSV files.
    Keeps all rows starting from the second row and preserves column order from the original file.
    Converts the date format to "%Y-%m-%d" for the 'date' column.
    Fills missing values with 0 and adds a new column 'money' initialized to 0.
    """
    # Columns to keep for all_0050
    '''
    columns_to_keep_all = [
        "2330", "2317", "2454", "2308", "2382", "2881", "2882", "2891", 
        "2303", "3711", "2412", "2886", "2357", "1216", "2884", "2885", 
        "3231", "2345", "2892", "2890", "3034", "5880", "2883", "2880", 
        "2002", "2327", "3008", "2379", "2603", "1101", "4938", "1303", 
        "3037", "2207", "2887", "2301", "6669", "3017", "3661", "1301", 
        "3045", "4904", "2395", "2912", "5876", "6446", "5871", "1326", 
        "1590", "6505", "Date", "0050", "9910", "2801", "2408", "2615",
        "2331", "2363", "2376", "2337", "2349", "2356", "2388", "2323",
        "2204", "1605", "2371", "2344", "6116", "2610", "2352", "8078",
        "2475", "3474", "8046", "9904", "2618", "2448", "2888", "1802",
        "1802", "2353", "1722", "3673", "2201", "2498", "2227", "1476",
        "2324", "2492", "2354", "3481", "2823", "2105", "2474", "2633",
        "1102", "1402", "8454", "6770", "2409", "6415", "2609"
    ]

    columns_to_keep_all = [
        "Date", "0050", "00881", "00757", "1216", "2882", "2881", "3231", "1319", "2308", "2382"
    ]
    '''
    columns_to_keep_all = [
        "Date", "0050", "2330", "1216", "2882", "2881", "3231", "1319", "2308", "2382", "1795"
    ]


    # Columns to keep for ETF_0050
    columns_to_keep_etf = ["0050", "Date"]

    # Input file path
    input_file = "nineNoETF_2010-2024.csv"  # Path to the input CSV file

    # Output file paths
    output_file_all = "all_nineNoETF_v1.csv"  # Path to the first output CSV file
    output_file_etf = "ETF_nineNoETF_v1.csv"  # Path to the second output CSV file

    try:
        # Read the input CSV file
        df = pd.read_csv(input_file)

        # Process all_0050
        existing_columns_all = [col for col in df.columns if col in columns_to_keep_all]
        missing_columns_all = [col for col in columns_to_keep_all if col not in df.columns]
        if missing_columns_all:
            print(f"Warning: The following columns were not found for all_0050: {missing_columns_all}")
        filtered_df_all = df[existing_columns_all]
        if 'Date' in filtered_df_all.columns:
            filtered_df_all = filtered_df_all.rename(columns={'Date': 'date'})
        if 'date' in filtered_df_all.columns:
            filtered_df_all.loc[:, 'date'] = pd.to_datetime(filtered_df_all['date']).dt.strftime('%Y-%m-%d')
        filtered_df_all = filtered_df_all.fillna(0)  # Fill missing values with 0
        filtered_df_all['money'] = 0  # Add 'money' column with default value 0
        filtered_df_all.to_csv(output_file_all, index=False)

        # Process ETF_0050
        existing_columns_etf = [col for col in df.columns if col in columns_to_keep_etf]
        missing_columns_etf = [col for col in columns_to_keep_etf if col not in df.columns]
        if missing_columns_etf:
            print(f"Warning: The following columns were not found for ETF_0050: {missing_columns_etf}")
        filtered_df_etf = df[existing_columns_etf]
        if 'Date' in filtered_df_etf.columns:
            filtered_df_etf = filtered_df_etf.rename(columns={'Date': 'date'})
        if 'date' in filtered_df_etf.columns:
            filtered_df_etf.loc[:, 'date'] = pd.to_datetime(filtered_df_etf['date']).dt.strftime('%Y-%m-%d')
        filtered_df_etf = filtered_df_etf.fillna(0)  # Fill missing values with 0
        filtered_df_etf.to_csv(output_file_etf, index=False)

        print(f"Processing complete! The results have been saved to: {output_file_all} and {output_file_etf}")
    except Exception as e:
        print(f"An error occurred: {e}")

# Run the function
filter_csv_columns()
