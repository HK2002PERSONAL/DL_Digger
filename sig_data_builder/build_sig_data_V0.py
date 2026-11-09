import os
import time

import pandas as pd
import numpy as np
import torch
import duckdb as db
import signatory

from joblib import Parallel, delayed

__START_TIME__, __END_TIME__ = 93000000, 150000000


def process_file(file_name):
    path = f"/data/cephfs/transaction/{file_name}"

    SecuCodeS = db.query(f"""SELECT DISTINCT security_code FROM read_parquet('{path}')""").fetchall()

    sig_feature = {}

    for i in SecuCodeS:
        _SecuCode = i[0]

        query = f''' 
        SELECT 
            transaction_time AS DataTime, 
            AVG(transaction_price) as Price, 
            SUM(transaction_volume) as Volume
        FROM read_parquet('{path}')
        WHERE security_code = '{_SecuCode}'
        GROUP BY transaction_time
        ORDER BY DataTime ASC
        '''
        raw_data = db.query(query).df()
        raw_data['Price'] = raw_data['Price'] / raw_data['Price'].iloc[0]
        raw_data['Volume'] = np.log1p(raw_data['Volume'].cumsum())
        raw_data['Volume'] = raw_data['Volume'] / raw_data['Volume'].iloc[-1]
        raw_data = torch.from_numpy(db.query(query).df().values)
        raw_data[:,0] = (raw_data[:,0] - __START_TIME__) / (__END_TIME__ - __START_TIME__)
        try:
            sig_feature[_SecuCode] = signatory.signature(raw_data.unsqueeze(0), depth=3).squeeze(0)
        except ValueError:
            continue

    pd.DataFrame(sig_feature, index=range(12)).T.to_parquet(os.path.join("data_raw", "sig_1d_raw", file_name))


if __name__ == "__main__":
    start = time.time()

    file_names = os.listdir("/data/cephfs/transaction/")
    file_names = [file_name for file_name in file_names if not file_name.startswith("2010") and file_name != "file_list.txt"]

    Parallel(n_jobs=32, verbose=5)(
        delayed(process_file)(file_name) for file_name in file_names
    )

    print("Task Ended in: ", time.time() - start, 's')