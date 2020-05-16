import pandas as pd
import whisk

def dataset_to_df():
    tweet= pd.read_csv(whisk.data_dir / 'raw/train.csv')
    test=pd.read_csv(whisk.data_dir / 'raw/test.csv')
    return [tweet,test]
