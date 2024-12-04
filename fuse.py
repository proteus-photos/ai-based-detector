import argparse
from utils.fusion import apply_fusion
import pandas as pd

if __name__=="__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--clip_csv"    , '-c', type=str, help="The path of the csv file containing clip logits", default="csvs/clip.csv")
    parser.add_argument("--other_csv"    , '-m', type=str, help="The path of the csv file containing other model logits")
    parser.add_argument("--type"    , '-t', type=str, help="Type of fusion", default="soft_or_prob")
    parser.add_argument("--out_csv", '-o', type=str, help="output stored in this csv")
    args = vars(parser.parse_args())
    
    clip_df =pd.read_csv(args['clip_csv'])
    other_df = pd.read_csv(args['other_csv'])
    table =clip_df[['path']].copy()
    for col in clip_df.columns:
        if col != 'path':
            cname = f'fusion_{col[12:]}'
            table[cname]=apply_fusion(pd.concat([clip_df[col],other_df['Corvi2023']],axis=1).values,args['type'],axis=-1)
    table.to_csv(args['out_csv'], index=False) 