import pandas as pd
import tqdm
import numpy as np
from tqdm import tqdm


def create_data(file_loc, target_loc):
    '''
    This function creates a data csv file with 1B records from a data with 50k records
    
    file_loc: location of the csv file to read
    target_loc: location to where augmented file to be placed
    
    '''
    data = pd.read_csv(file_loc)
    data = data[['Time', 'Amount', 'Class']]

    for i in tqdm(range(20064)):
        #a = (np.random.random(len(data)) + 0.5)
        #data['UpdatedAmount'] = ((data.Amount * a).round(2))
        data['UpdatedAmount'] = data['Amount']
        if i == 0:
            data[['Time', 'UpdatedAmount', 'Class']].to_csv(f'{target_loc}/Credit1B.csv', index=False)
        else:
            data[['Time', 'UpdatedAmount', 'Class']].to_csv(f'{target_loc}/Credit1B.csv', mode='a', index=False, header=False)