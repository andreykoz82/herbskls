# %%
import os
from shutil import copyfile
from pathlib import Path
import time
import re
import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings('ignore')

# os.chdir('/home/andreykozinov/YandexGPU/HerbsKLS')

def change_file_extension(filename):
    name, ext = os.path.splitext(filename)
    if ext == ".JPG":
        os.rename(filename, name + ".jpg")


def rename_files(filename, directory):
    if Path().joinpath('raw_data', '2021') == Path(directory):
        return ''.join(re.findall(r'\d', filename)) + '_2021'
    if Path().joinpath('raw_data', '2020', 'samples') == Path(directory):
        return ''.join(re.findall(r'\d', filename)) + '_2020_sample'
    if Path().joinpath('raw_data', '2020', 'food') == Path(directory):
        return ''.join(re.findall(r'\d', filename)) + '_2020_food'
    if Path().joinpath('raw_data', '2020') == Path(directory):
        return ''.join(re.findall(r'\d', filename)) + '_2020'
    if Path().joinpath('raw_data', '2019') == Path(directory):
        return ''.join(re.findall(r'\d', filename)) + '_2019'


os.makedirs(Path().joinpath('data'))
time.sleep(3)
temp_folder = 'data/'

for subdir, dirs, files in os.walk('raw_data'):
    for file in files:
        src_file = Path().joinpath(subdir, file)
        dest_file = Path().joinpath(temp_folder, file)
        copyfile(src_file, dest_file)

        new_file_name = rename_files(filename=temp_folder + file, directory=subdir)
        name, ext = os.path.splitext(temp_folder + file)
        os.rename(name + ext, temp_folder + new_file_name + ext)
        change_file_extension(temp_folder + new_file_name + ext)

quality_control = pd.read_excel('certificates/certificates_2018_2019.xlsx')
quality_control.data = pd.to_datetime(quality_control.data, dayfirst=True).dt.year.apply(str)
quality_control.number = quality_control.number.str[-3:].str.lstrip('0')

file_name_table = []
for idx, row in quality_control.iterrows():
    if row[3] == 'drug':
        drug = row[1] + '_' + row[0]
        file_name_table.append(drug)
    else:
        other = row[1] + '_' + row[0] + '_' + row[3]
        file_name_table.append(other)

quality_control['file_name'] = file_name_table
quality_control = quality_control[['file_name', 'item']]

temp_folder = 'data/'
files_list = [name.split('.')[0] for name in os.listdir(Path().joinpath(temp_folder))]
print("Количество фотографий = ", len(files_list))

file_df = pd.DataFrame(files_list).rename(columns={0: 'file_name'})
quality_control = quality_control.merge(file_df, how='right', on='file_name')
quality_control.dropna(inplace=True)
print("Количество фото после объединения", quality_control.shape[0])


def train_test_split(df, proportion):
    # Создаем колонку number с количеством изображений по каждому наименованию для train
    split = (proportion * df.groupby('item').count()).reset_index()
    split['file_name'] = split['file_name'].apply(np.ceil)
    split = split.rename(columns={'file_name': 'train_num'})
    df = pd.merge(df, split, how='left', on='item')
    df['split'] = 0

    # Если количество изображений по наименованию составляет 1, то дублируем эту строку
    single_group = df.loc[df['train_num'] == 1]
    df = df.append(single_group, ignore_index=True)
    df = df.sort_values(by='item').reset_index(drop='True')

    # Создаем колонку split со значениями train и validation
    # Train = количество изображений по каждой категории * процент разделения
    # Validation = количество изображений по каждой категории - Train
    counter = 0
    for i in range(0, df.shape[0] - 1):
        if df['item'][i] == df['item'][i + 1]:
            if counter < df['train_num'][i]:
                df['split'][i] = 'train'
                counter += 1
            else:
                df['split'][i] = 'test'
                counter += 1
        else:
            df['split'][i] = 'test'
            counter = 0
    # Присоединяем к номеру изображения расширение jpg
    df['file_name'] = df['file_name'] + '.jpg'

    return df.drop('train_num', 1)


quality_control = train_test_split(quality_control, 0.7)
quality_control.iloc[-1]['split'] = 'test'
files_list = [name.lower() for name in os.listdir(Path().joinpath(temp_folder))]

os.makedirs(Path().joinpath(temp_folder, 'dataset'))
os.makedirs(Path().joinpath(temp_folder, 'dataset', 'train'))
os.makedirs(Path().joinpath(temp_folder, 'dataset', 'val'))
for row in quality_control.iterrows():
    if row[1][0] in files_list:
        if row[1][2] == 'train':
            if os.path.exists(Path().joinpath(temp_folder, 'dataset', 'train', row[1][1])):
                copyfile(Path().joinpath(temp_folder, row[1][0]),
                         Path().joinpath(temp_folder, 'dataset', 'train', row[1][1], row[1][0]))
            else:
                os.makedirs(Path().joinpath(temp_folder, 'dataset', 'train', row[1][1]))
                copyfile(Path().joinpath(temp_folder, row[1][0]),
                         Path().joinpath(temp_folder, 'dataset', 'train', row[1][1], row[1][0]))
        if row[1][2] == 'test':
            if os.path.exists(Path().joinpath(temp_folder, 'dataset', 'val', row[1][1])):
                copyfile(Path().joinpath(temp_folder, row[1][0]),
                         Path().joinpath(temp_folder, 'dataset', 'val', row[1][1], row[1][0]))
            else:
                os.makedirs(Path().joinpath(temp_folder, 'dataset', 'val', row[1][1]))
                copyfile(Path().joinpath(temp_folder, row[1][0]),
                         Path().joinpath(temp_folder, 'dataset', 'val', row[1][1], row[1][0]))
