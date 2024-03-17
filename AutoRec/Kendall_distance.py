import pandas as pd

def kendall_distance_with_penalty(seq1, seq2, id_seq1, id_seq2, rating_seq1, rating_seq2, p=0.1):
    """
    Calculate Kendall distance with penalty between two sequences based on their ratings.

    Parameters:
    - seq1 (pandas DataFrame): First sequence dataframe.
    - seq2 (pandas DataFrame): Second sequence dataframe.
    - id_seq1 (str): Name of the ID column in the first sequence dataframe.
    - id_seq2 (str): Name of the ID column in the second sequence dataframe.
    - rating_seq1 (str): Name of the column containing ratings in the first sequence dataframe.
    - rating_seq2 (str): Name of the column containing ratings in the second sequence dataframe.
    - p (float, optional): Penalty factor for tied rankings in one sequence but not the other.
                           Should be between 0 and 1. Default is 0.1.

    Returns:
    - kendall (float): Kendall distance with penalty between the two sequences.
    
    References:
        Fagin, R., Kumar, R., Mahdian, M., Sivakumar, D., & Vee, E. (2006). Comparing partial rankings. 
        SIAM Journal on Discrete Mathematics, 20(3), 628-648.
        
    The Kendall distance with penalty measures the similarity between two sequences
    based on their rankings. It counts the number of pairwise disagreements in rankings
    with an additional penalty applied if items are tied in one sequence but not in the other.

    For tied rankings in both sequences, no penalty is applied.
    For disagreements in rankings where both sequences order the items the same way, no penalty is applied.
    For disagreements in rankings where sequences order the items differently, a penalty of 1 is applied.
    For tied rankings in one sequence but not in the other, a penalty specified by parameter p is applied.

    """
    # Rename rating_seq2 if both rating columns have the same name
    if rating_seq1 == rating_seq2:
        seq2.rename(columns={rating_seq2: 'Rating_Seq2'}, inplace=True)
        rating_seq2 = 'Rating_Seq2'
    
    # Rename ID columns for merge
    seq1.rename(columns={id_seq1: 'ID'}, inplace=True)
    seq2.rename(columns={id_seq2: 'ID'}, inplace=True)

    # Remove items not in both sequences
    merged_df = pd.merge(seq1, seq2, on='ID', how='inner')
    
    # Generate rankings for both ranking columns
    merged_df['Ranking_seq1'] = merged_df[rating_seq1].rank(method='min')
    merged_df['Ranking_seq2'] = merged_df[rating_seq2].rank(method='min')
    
    # Initialize Kendall distance score
    kendall = 0

    # Iterate over each possible item combination
    for i in merged_df['ID']:
        for j in merged_df['ID']:
            if j > i:
                comparison_1 = merged_df.loc[merged_df['ID']==i, 'Ranking_seq1'].values[0] - merged_df.loc[merged_df['ID']==j, 'Ranking_seq1'].values[0]
                comparison_2 = merged_df.loc[merged_df['ID']==i, 'Ranking_seq2'].values[0] - merged_df.loc[merged_df['ID']==j, 'Ranking_seq2'].values[0]
                
                # Case 1: items are tied in both rankings
                if comparison_1 == 0 and comparison_2 == 0:
                    kendall += 0
                # Case 2: items are not tied in both ratings
                # Both sequences ordered the items the same way
                elif (comparison_1 < 0 and comparison_2 < 0) or (comparison_1 > 0 and comparison_2 > 0):
                    kendall += 0
                # The sequences ordered the items differently
                elif (comparison_1 < 0 and comparison_2 > 0) or (comparison_1 > 0 and comparison_2 < 0):
                    kendall += 1
                # Case 3: tied in one sequence, but not in the other
                elif (comparison_1 == 0 and comparison_2 != 0) or (comparison_1 != 0 and comparison_2 == 0):
                    kendall += p
    
    return kendall
