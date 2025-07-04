import pandas as pd
import numpy as np

def compute_stats():
    pass

def compute_merged_file():
    pass

def main():
    FILENAMES= [
        'EVAL_1',
        'EVAL_2',
        'OUT_SCAL_1',
        'OUT_SCAL_2',
        'ALL_LABELED',
        'IN_TOP_MOD_1',
    ]

    # ========== Working on SCAL_1 ==========
    df_scal_1 = pd.read_csv('OUT_SCAL_1.csv')
    df_scal_1['ID'] = df_scal_1['ID'].astype(str)  # Ensure ID is treated as string
    print(f"SCAL_1: {df_scal_1.shape}\t\t\t(rel = {np.sum(df_scal_1['Score']==1)} - sugg = {np.sum(df_scal_1['Score']!=1):,})")
    assert df_scal_1.shape[0] == len(set(df_scal_1['ID'])), "There are duplicate IDs in OUT_SCAL_1.csv"


    # ========== Working on SCAL_2 ==========
    df_scal_2 = pd.read_csv('OUT_SCAL_2.csv')
    df_scal_2['ID'] = df_scal_2['ID'].astype(str)  # Ensure ID is treated as string
    print(f"SCAL_2: {df_scal_2.shape}\t\t\t(rel = {np.sum(df_scal_2['Score']==1)} - sugg = {np.sum(df_scal_2['Score']!=1):,})")
    assert df_scal_2.shape[0] == len(set(df_scal_2['ID'])), "There are duplicate IDs in OUT_SCAL_2.csv"
                        
    # ========== Working on EVAL_1 ==========
    df_eval_1 = pd.read_csv('EVAL_1.csv')
    df_eval_1['ID'] = df_eval_1['ID'].astype(str)  # Ensure ID is treated as string
    print(f"EVAL_1: {df_eval_1.shape}\t\t\t(rel = {np.sum(df_eval_1['Label']=='Relevant')} - irrel = {np.sum(df_eval_1['Label']=='Irrelevant'):,})")
    assert df_eval_1.shape[0] == len(set(df_eval_1['ID'])), "There are duplicate IDs in EVAL_1.csv"

    # ========== Working on EVAL_2 ==========
    df_eval_2 = pd.read_csv('EVAL_2.csv')
    df_eval_2['ID'] = df_eval_2['ID'].astype(str)  # Ensure ID is treated as string
    print(f"EVAL_2: {str(df_eval_2.shape):8}\t\t\t(rel = {np.sum(df_eval_2['Label']=='Relevant')} - irrel = {np.sum(df_eval_2['Label']=='Irrelevant'):,})")
    assert df_eval_2.shape[0] == len(set(df_eval_2['ID'])), "There are duplicate IDs in EVAL_2.csv"

    # ========== Working on ALL_LABELED ==========
    df_all_labeled = pd.read_csv('ALL_LABELED.csv')
    df_all_labeled['ID'] = df_all_labeled['ID'].astype(str)  # Ensure ID is treated as string
    print(f"ALL_LABELED (no repeat): {df_all_labeled.shape}\t(rel = {np.sum(df_all_labeled['Labeled (corrected)']=='Relevant')} - irrel = {np.sum(df_all_labeled['Labeled (corrected)']=='Irrelevant'):,})")
    assert df_all_labeled.shape[0] == len(set(df_all_labeled['ID'])), "There are duplicate IDs in ALL_LABELED.csv"

    relevant_ids = set(df_all_labeled[df_all_labeled['Labeled (corrected)']=='Relevant']['ID'])
    all_labeled_ids = set(df_all_labeled['ID'])
    print(f"Number of relevant IDs in ALL_LABELED: {len(relevant_ids)}")

    print(f'Number of actual relevant in SCAL_1 relevant: {np.sum(df_scal_1[df_scal_1["Score"]==1]["ID"].isin(relevant_ids))}')
    print(f'Number of actual relevant in SCAL_2 relevant: {np.sum(df_scal_2[df_scal_2["Score"]==1]["ID"].isin(relevant_ids))}')


    print(f"Number of labeled items in EVAL_1: {np.sum(df_eval_1['ID'].isin(all_labeled_ids))}")
    print(f"Number of labeled items in EVAL_2: {np.sum(df_eval_2['ID'].isin(all_labeled_ids))}")

    print(f"Number of relevant in EVAL_1: {np.sum(df_eval_1[df_eval_1['Label']=='Relevant']['ID'].isin(relevant_ids))}")
    print(f"Number of relevant in EVAL_2: {np.sum(df_eval_2[df_eval_2['Label']=='Relevant']['ID'].isin(relevant_ids))}")
    

    # Now, the 17,014 and 8,115
    suggestion_in_scal_1 = df_scal_1[df_scal_1['Score']!=1]['ID']
    suggestion_in_scal_2 = df_scal_2[df_scal_2['Score']!=1]['ID']

    print(f"Number of suggestion in SCAL_1 (removing labeled): {len(set(suggestion_in_scal_1).difference(all_labeled_ids)):,}")
    print(f"Number of suggestion in SCAL_2 (removing labeled): {len(set(suggestion_in_scal_2).difference(all_labeled_ids)):,}")


    in_top_mod_1_df = pd.read_csv('IN_TOP_MOD_1.csv')
    in_top_mod_1_ids = set(in_top_mod_1_df['ID'].astype(str))

    print(f"IN_TOP_MOD_1: {in_top_mod_1_df.shape}")

    print(f"Number of labeled in IN_TOP_MOD_1: {len(set(in_top_mod_1_ids).intersection(all_labeled_ids))}")
    print(f"Number of relevant in IN_TOP_MOD_1: {len(set(in_top_mod_1_ids).intersection(relevant_ids))}")
    print(f"Number of suggestion in IN_TOP_MOD_1: {len(set(in_top_mod_1_ids).difference(all_labeled_ids)):,}")


    out_top_mod_2_df = pd.read_csv('OUT_TOP_MOD_2.csv')
    out_top_mod_2_ids = set(out_top_mod_2_df['ID'].astype(str))
    print(f"OUT_TOP_MOD_2: {out_top_mod_2_df.shape}")
    print(f"Number of labeled in OUT_TOP_MOD_2: {len(set(out_top_mod_2_ids).intersection(all_labeled_ids))}")
    print(f"Number of relevant in OUT_TOP_MOD_2: {len(set(out_top_mod_2_ids).intersection(relevant_ids))}")
    print(f"Number of suggestion in OUT_TOP_MOD_2: {len(set(out_top_mod_2_ids).difference(all_labeled_ids)):,}")

    # ========== Final Merged File ==========
    scal_suggestion_ids = set(df_scal_1['ID']).intersection(set(df_scal_2['ID'])).difference(all_labeled_ids)
    print(f"Number of suggestion IDs in both SCAL_1 and SCAL_2: {len(scal_suggestion_ids):,}")

    final_suggestion_ids = scal_suggestion_ids.intersection(in_top_mod_1_ids).intersection(out_top_mod_2_ids).difference(all_labeled_ids)

    print(f"Number of final suggestions (removing labeled): {len(final_suggestion_ids):,}")

    print(f"Number of final relevant: {len((relevant_ids)):,}")

    assert np.sum(df_scal_1['ID'].isin(final_suggestion_ids))==len(final_suggestion_ids), f"All IDs in SCAL_1 should be in final suggestions, but found {np.sum(df_scal_1['ID'].isin(final_suggestion_ids))} out of {df_scal_1.shape[0]}"
    assert len(relevant_ids.intersection(set(df_scal_1['ID'])))==len(relevant_ids), f"All labeled IDs should be in final suggestions, but found {len(relevant_ids.intersection(set(df_scal_1['ID'])))} out of {len(all_labeled_ids)}"

    # attach topic:
    id2topic = {}
    for id_, topic_no in zip(out_top_mod_2_df['ID'], out_top_mod_2_df['Topic No.']):
        id2topic[str(id_)] = topic_no
        
    # combine suggestions and relevant:
    suggestions_df = df_scal_1[[id_ in final_suggestion_ids for id_ in df_scal_1['ID']]].copy()
    suggestions_df['Topic No.'] = suggestions_df['ID'].apply(lambda x: id2topic.get(str(x), 'N/A'))
    suggestions_df.sort_values(by='Topic No.', inplace=True)

    relevant_df = df_scal_1[[id_ in relevant_ids for id_ in df_scal_1['ID']]].copy()
    relevant_df['Topic No.'] = relevant_df['ID'].apply(lambda x: id2topic.get(str(x), 'N/A'))
    relevant_df.sort_values(by='Topic No.', inplace=True)

    final_df = pd.concat([relevant_df, suggestions_df], ignore_index=True)
    final_df.drop(columns=['Score'], inplace=True)  # Drop the Score column as it's not needed in the final output
    final_df['Rel or Sugg'] = ['Rel']* len(relevant_df) + ['Sugg']*len(suggestions_df)


    final_df['URL'] = final_df['ID'].apply(lambda x: f"https://www.proquest.com/docview/{x}")

    final_df.to_csv('FINAL_MERGED_FILE.csv', index=False)


    print('Done!')
    
    
if __name__ == "__main__":
    main()