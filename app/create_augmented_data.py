import pandas as pd
import tqdm


def create_data(file_loc):
    data = pd.read_csv(file_loc)
    data = data[['Time', 'Amount', 'Class']]
    data = pd.concat([data]*4)
    data = data.reset_index(drop=True)

    for i in range(50400):
        data.to_parquet(f'data/scaled/part.{i}.parquet', compression='zstd',)