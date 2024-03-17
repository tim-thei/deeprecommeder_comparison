import pandas as pd

def rec_split(df, user, time=None, train_share=0.6, val_share=0.3):
    
    """
    Splits a DataFrame into training, validation, and test sets with data of each user in training, validation and test data.

    Parameters:
    - df (pd.DataFrame): The input DataFrame containing user-specific data.
    - user (str): The name of the column representing user IDs.
    - time (str or None, optional): The name of the column representing timestamps. If provided, the data is sorted in chronological order. If None, the data is shuffled.
    - train_share (float, optional): The proportion of data to allocate to the training set. Default is 0.6.
    - val_share (float, optional): The proportion of data to allocate to the validation set. Default is 0.3.

    Returns:
    - train_df (pd.DataFrame): The training set containing train_share of the data for each user.
    - val_df (pd.DataFrame): The validation set containing val_share of the data for each user.
    - test_df (pd.DataFrame): The test set containing the remaining data for each user.

    Note: The sum of train_share and val_share should be less than or equal to 1.
    """
    
    train = []
    val = []
    test = []

    u_IDs = df[user].unique()

    for u in u_IDs:
        data=df[df[user] == u]
        
        if time != None: # sort in chronological order
            data = data.sort_values(by=time, ascending=True)
            data.drop(time, axis = 1, inplace = True)
        
        else: # shuffel the data
            data= data.sample(frac = 1)        

        # Determine the sizes of each split
        item_count = len(data)
        train_size = int(train_share * item_count)
        val_size = int(val_share * item_count)
        test_size = item_count - train_size - val_size

        # Split the data
        train_data = data.head(train_size)
        val_data = data.iloc[train_size: train_size + val_size]
        test_data = data.tail(test_size)

        # Append to the corresponding lists
        train.append(train_data)
        val.append(val_data)
        test.append(test_data)

    # Concatenate the lists to obtain the final DataFrames
    train_df = pd.concat(train, ignore_index=True)
    val_df = pd.concat(val, ignore_index=True)
    test_df = pd.concat(test, ignore_index=True)

    """
    # Save DataFrames to Parquet files
    train_df.to_parquet(output_path + '_train.parquet', index=False)
    val_df.to_parquet(output_path + '_val.parquet', index=False)
    test_df.to_parquet(output_path + '_test.parquet', index=False)
    """

    #return dataframes
    return train_df, val_df, test_df