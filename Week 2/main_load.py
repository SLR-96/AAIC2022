from tensorflow.keras.models import load_model
import numpy as np
import pandas as pd

path = './My Model'

x_min = np.load(path + '/x_min.npy')
x_max = np.load(path + '/x_max.npy')

my_model = load_model(path)


train_file = pd.read_csv('./MCI-RD-aaic-UIUF/MCIRD_aaic2021_train.csv', index_col='subscriber_ecid')
train_file.insert(train_file.shape[1] - 1, 'data_usage_volume', train_file.pop('data_usage_volume'))

week1_file = pd.read_csv('./MCI-RD-aaic-UIUF/MCIRD_aaic2021_test_week1_with_target.csv', index_col='subscriber_ecid')
week1_file.insert(week1_file.shape[1] - 1, 'data_usage_volume', week1_file.pop('data_usage_volume'))

week2_file = pd.read_csv('./MCI-RD-aaic-UIUF/MCIRD_aaic2021_test_week2.csv', index_col='subscriber_ecid')
week2_file.insert(week2_file.shape[1], 'data_usage_volume', np.zeros(week2_file.shape[0]))

people = np.unique(week2_file.index)

final_df = None
for person in people:
    person_train = train_file.loc[person]
    person_week1 = week1_file.loc[person]
    person_week2 = week2_file.loc[person]
    days = person_week2.shape[0]
    person_data = pd.concat([person_train, person_week1, person_week2]).drop('day', axis=1).values
    data_dim = person_data.shape
    for i in range(data_dim[0]):
        for j in range(data_dim[1]):
            if np.isnan(person_data[i, j]):
                person_data[i, j] = np.nanmean(person_data[:, j])
    for i in reversed(range(days)):
        if i == 0:
            sequence = person_data.copy()
        else:
            sequence = person_data.copy()[:-i]
        x = (sequence - x_min) / (x_max - x_min)
        person_data[-i-1, -1] = float(my_model.predict(x[np.newaxis, :, :]))
    final_person_data = person_data[-days:, -1]
    person_week2.reset_index(level=0, inplace=True)
    person_week2['data_usage_volume'] = final_person_data
    person_df = person_week2[['day', 'subscriber_ecid', 'data_usage_volume']]
    if final_df is None:
        final_df = person_df
    else:
        final_df = pd.concat([final_df, person_df])

final_df.to_csv('./week2_results.csv', index=False)

