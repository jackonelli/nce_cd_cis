# Adapted from https://github.com/conormdurkan/autoregressive-energy-machines

import numpy as np
import os
import pandas as pd
import shutil
import sys
import tarfile

import h5py

from collections import Counter
from tqdm import tqdm
from urllib import request


############
# DOWNLOAD #
############


def download_and_extract(data_dir):

    filename = os.path.join(data_dir, 'data.tar.gz')

    def reporthook(tbar):
        """
        tqdm report hook for data download.
        From https://github.com/tqdm/tqdm/blob/master/examples/tqdm_wget.py
        """
        last_block = [0]

        def update_to(block=1, block_size=1, tbar_size=None):
            if tbar_size is not None:
                tbar.total = tbar_size
            tbar.update((block - last_block[0]) * block_size)
            last_block[0] = block

        return update_to

    with tqdm(unit='B', unit_scale=True, unit_divisor=1024, miniters=1) as tbar:
        tbar.set_description('Downloading datasets...')
        request.urlretrieve(
            url='https://zenodo.org/record/1161203/files/data.tar.gz?download=1',
            filename=filename,
            reporthook=reporthook(tbar)
        )
        tbar.set_description('Finished downloading.')

    print('\nExtracting datasets...')
    with tarfile.open(filename, 'r:gz') as file:
        file.extractall(path=data_dir)
    print('Finished extraction.\n')

    print('Removing zipped data...')
    os.remove(filename)
    print('Zipped data removed.\n')

    response = input('Do you wish to delete CIFAR-10? [y/n]')
    if response in ['y', '']:
        shutil.rmtree(os.path.join(data_dir, 'data/cifar10'))
        print('CIFAR-10 removed.\n')
    elif response == 'n':
        print('CIFAR-10 not removed.\n')
    else:
        print('Response not understood. CIFAR-10 not removed.\n')


def download_data():
    data_root = get_data_root()
    if not os.path.isdir(os.path.join(data_root, 'data')):
        query = (
            "> UCI, BSDS300, and MNIST data not found.\n"
            "> The zipped download is 817MB in size, and 1.6GB once unzipped.\n"
            "> The download includes CIFAR-10, although it is not used in the paper.\n"
            "> After extraction, this script will delete the zipped download.\n"
            "> You will also be able to specify whether to delete CIFAR-10.\n"
            "> Do you wish to download the data? [y/n]"
        )
        response = input(query)
        if response in ['y', '']:
            download_and_extract(data_root)
        elif response == 'n':
            sys.exit()
        else:
            print('Response not understood.')
            sys.exit()


#########
# POWER #
#########


def load_power(scaling):
    def load_data():
        file = os.path.join(get_data_root(), 'data', 'power/data.npy')
        try:
            return np.load(file)
        except FileNotFoundError:
            download_data()
            return np.load(file)

    def load_data_split_with_noise():
        rng = np.random.RandomState(42)

        data = load_data()
        rng.shuffle(data)
        N = data.shape[0]

        data = np.delete(data, 3, axis=1)
        data = np.delete(data, 1, axis=1)
        ############################
        # Add noise
        ############################
        # global_intensity_noise = 0.1*rng.rand(N, 1)
        voltage_noise = 0.01 * rng.rand(N, 1)
        # grp_noise = 0.001*rng.rand(N, 1)
        gap_noise = 0.001 * rng.rand(N, 1)
        sm_noise = rng.rand(N, 3)
        time_noise = np.zeros((N, 1))
        # noise = np.hstack((gap_noise, grp_noise, voltage_noise, global_intensity_noise, sm_noise, time_noise))
        # noise = np.hstack((gap_noise, grp_noise, voltage_noise, sm_noise, time_noise))
        noise = np.hstack((gap_noise, voltage_noise, sm_noise, time_noise))
        data = data + noise

        N_test = int(0.1 * data.shape[0])
        data_test = data[-N_test:]
        data = data[0:-N_test]
        N_validate = int(0.1 * data.shape[0])
        data_validate = data[-N_validate:]
        data_train = data[0:-N_validate]

        return data_train, data_validate, data_test

    def load_data_normalised(scaling):
        data_train, data_validate, data_test = load_data_split_with_noise()
        #data = np.vstack((data_train, data_validate))
        
        if scaling == 'standardize':
            print("Standardizing data")
            mu = data_train.mean(axis=0)  # data.mean(axis=0)
            s = data_train.std(axis=0)  # data.std(axis=0)
            data_train = (data_train - mu) / s
            data_validate = (data_validate - mu) / s
            data_test = (data_test - mu) / s

        elif scaling == 'normalize':
            print("Normalizing data")
            min_val = data_train.min(axis=0)  # data.mean(axis=0)
            max_val = data_train.max(axis=0)  # data.std(axis=0)
            data_train = (data_train - min_val) / (max_val - min_val)
            data_validate = (data_validate - min_val) / (max_val - min_val)
            data_test = (data_test - min_val) / (max_val - min_val)

        return data_train, data_validate, data_test

    return load_data_normalised(scaling)


def preprocess_and_save_power(scaling='standardize'):
    train, val, test = load_power(scaling)
    splits = (
        ('train', train),
        ('val', val),
        ('test', test)
    )
    for split in splits:
        name, data = split
        
        if scaling == 'standardize':
            file = os.path.join(get_data_root(), 'data', 'power/{}.npy'.format(name))
        
        elif scaling == 'normalize':
        
            file_path = os.path.join(get_data_root(), 'data_norm', 'power')
            if not os.path.exists(file_path):
                os.makedirs(file_path)
        
            file = os.path.join(file_path, '{}.npy'.format(name))
        
        print("Saving data to: " + str(file))
        np.save(file, data)


#######
# GAS #
#######

def load_gas(scaling):
    def load_data(file):
        try:
            data = pd.read_pickle(file)
        except FileNotFoundError:
            download_data()
            data = pd.read_pickle(file)
        data.drop("Meth", axis=1, inplace=True)
        data.drop("Eth", axis=1, inplace=True)
        data.drop("Time", axis=1, inplace=True)
        return data

    def get_correlation_numbers(data):
        C = data.corr()
        A = C > 0.98
        B = A.sum(axis=1)
        return B

    def load_data_and_clean(file):
        data = load_data(file)
        B = get_correlation_numbers(data)

        # TODO: this feels like something that should be done only on training data?
        while np.any(B > 1):
            col_to_remove = np.where(B > 1)[0][0]
            col_name = data.columns[col_to_remove]
            data.drop(col_name, axis=1, inplace=True)
            B = get_correlation_numbers(data)
        #data = (data - data.mean()) / data.std()

        return data.values

    def load_data_and_clean_and_split(file, scaling):
        data = load_data_and_clean(file)
        N_test = int(0.1 * data.shape[0])
        data_test = data[-N_test:]
        data_train = data[0:-N_test]
        N_validate = int(0.1 * data_train.shape[0])
        data_validate = data_train[-N_validate:]
        data_train = data_train[0:-N_validate]
        
            
        if scaling == 'standardize':
            print("Standardizing data")
            mu = data_train.mean(axis=0)  # data.mean(axis=0)
            s = data_train.std(axis=0)  # data.std(axis=0)
            data_train = (data_train - mu) / s
            data_validate = (data_validate - mu) / s
            data_test = (data_test - mu) / s

        elif scaling == 'normalize':
            print("Normalizing data")
            min_val = data_train.min(axis=0)  # data.mean(axis=0)
            max_val = data_train.max(axis=0)  # data.std(axis=0)
            data_train = (data_train - min_val) / (max_val - min_val)
            data_validate = (data_validate - min_val) / (max_val - min_val)
            data_test = (data_test - min_val) / (max_val - min_val)
        
        # TODO: Not sure if this is what we want to do (and this might change convergence times?)
        #data = (data - data.mean()) / data.std()
       

        return data_train, data_validate, data_test

    file = os.path.join(get_data_root(), 'data', 'gas/ethylene_CO.pickle')
    return load_data_and_clean_and_split(file, scaling)


def preprocess_and_save_gas(scaling='standardize'):
    train, val, test = load_gas(scaling)
    splits = (
        ('train', train),
        ('val', val),
        ('test', test)
    )
    for split in splits:
        name, data = split
        
        if scaling == 'standardize':
            file = os.path.join(get_data_root(), 'data', 'gas/{}.npy'.format(name))
        
        elif scaling == 'normalize':
        
            file_path = os.path.join(get_data_root(), 'data_norm', 'gas')
            if not os.path.exists(file_path):
                os.makedirs(file_path)
        
            file = os.path.join(file_path, '{}.npy'.format(name))
        
        print("Saving data to: " + str(file))
        np.save(file, data)


###########
# HEPMASS #
###########


def load_hepmass(scaling):

    def load_data(path):
        try:
            data_train = pd.read_csv(
                filepath_or_buffer=os.path.join(path, "1000_train.csv"),
                index_col=False)
            data_test = pd.read_csv(
                filepath_or_buffer=os.path.join(path, "1000_test.csv"),
                index_col=False)
        except FileNotFoundError:
            download_data()
            data_train = pd.read_csv(
                filepath_or_buffer=os.path.join(path, "1000_train.csv"),
                index_col=False)
            data_test = pd.read_csv(
                filepath_or_buffer=os.path.join(path, "1000_test.csv"),
                index_col=False)
        return data_train, data_test

    def load_data_no_discrete(path):
        """
        Loads the positive class examples from the first 10 percent of the dataset.
        """
        data_train, data_test = load_data(path)

        # Gets rid of any background noise examples i.e. class label 0.
        data_train = data_train[data_train[data_train.columns[0]] == 1]
        data_train = data_train.drop(data_train.columns[0], axis=1)
        data_test = data_test[data_test[data_test.columns[0]] == 1]
        data_test = data_test.drop(data_test.columns[0], axis=1)
        # Because the data_ set is messed up!
        data_test = data_test.drop(data_test.columns[-1], axis=1)

        return data_train, data_test

    def load_data_no_discrete_normalised(path, scaling):

        data_train, data_test = load_data_no_discrete(path)

        N = data_train.shape[0]
        N_validate = int(N * 0.1)
        data_validate = data_train[-N_validate:]
        data_train = data_train[0:-N_validate]

        if scaling == 'standardize':
            print("Standardizing data")
            mu = data_train.mean(axis=0)  # data.mean(axis=0)
            s = data_train.std(axis=0)  # data.std(axis=0)
            data_train = (data_train - mu) / s
            data_validate = (data_validate - mu) / s
            data_test = (data_test - mu) / s

        elif scaling == 'normalize':
            print("Normalizing data")
            min_val = data_train.min(axis=0)  # data.mean(axis=0)
            max_val = data_train.max(axis=0)  # data.std(axis=0)
            data_train = (data_train - min_val) / (max_val - min_val)
            data_validate = (data_validate - min_val) / (max_val - min_val)
            data_test = (data_test - min_val) / (max_val - min_val)

        return data_train, data_validate, data_test


    def load_data_no_discrete_normalised_as_array(path, scaling):

        data_train, data_validate, data_test = load_data_no_discrete_normalised(path, scaling)
        
        data_train, data_validate, data_test = data_train.values, data_validate.values, data_test.values

        i = 0
        # Remove any features that have too many re-occurring real values.
        features_to_remove = []
        for feature in data_train.T:
            c = Counter(feature)
            max_count = np.array([v for k, v in sorted(c.items())])[0]
            if max_count > 5:
                features_to_remove.append(i)
            i += 1
        data_train = data_train[:, np.array(
            [i for i in range(data_train.shape[1]) if i not in features_to_remove])]
        data_validate = data_validate[:, np.array(
            [i for i in range(data_validate.shape[1]) if i not in features_to_remove])]
        data_test = data_test[:, np.array(
            [i for i in range(data_test.shape[1]) if i not in features_to_remove])]

        return data_train, data_validate, data_test

    path = os.path.join(get_data_root(), 'data', 'hepmass')
    
    return load_data_no_discrete_normalised_as_array(path, scaling)
    

def preprocess_and_save_hepmass(scaling='standardize'):
    train, val, test = load_hepmass(scaling)
    splits = (
        ('train', train),
        ('val', val),
        ('test', test)
    )
    for split in splits:
        name, data = split
        
        if scaling == 'standardize':
            file = os.path.join(get_data_root(), 'data', 'hepmass/{}.npy'.format(name))
        
        elif scaling == 'normalize':
        
            file_path = os.path.join(get_data_root(), 'data_norm', 'hepmass')
            if not os.path.exists(file_path):
                os.makedirs(file_path)
        
            file = os.path.join(file_path, '{}.npy'.format(name))
                        
        print("Saving data to: " + str(file))
        np.save(file, data)
        

#############
# MINIBOONE #
#############


def load_miniboone(scaling):
    def load_data(root_path):
        # NOTE: To remember how the pre-processing was done.
        # data_ = pd.read_csv(root_path, names=[str(x) for x in range(50)], delim_whitespace=True)
        # print data_.head()
        # data_ = data_.as_matrix()
        # # Remove some random outliers
        # indices = (data_[:, 0] < -100)
        # data_ = data_[~indices]
        #
        # i = 0
        # # Remove any features that have too many re-occuring real values.
        # features_to_remove = []
        # for feature in data_.T:
        #     c = Counter(feature)
        #     max_count = np.array([v for k, v in sorted(c.iteritems())])[0]
        #     if max_count > 5:
        #         features_to_remove.append(i)
        #     i += 1
        # data_ = data_[:, np.array([i for i in range(data_.shape[1]) if i not in features_to_remove])]
        # np.save("~/data_/miniboone/data_.npy", data_)

        try:
            data = np.load(root_path)
        except FileNotFoundError:
            download_data()
            data = np.load(root_path)
        N_test = int(0.1 * data.shape[0])
        data_test = data[-N_test:]
        data = data[0:-N_test]
        N_validate = int(0.1 * data.shape[0])
        data_validate = data[-N_validate:]
        data_train = data[0:-N_validate]

        return data_train, data_validate, data_test

    def load_data_normalised(root_path, scaling):
        data_train, data_validate, data_test = load_data(root_path)
        # data = np.vstack((data_train, data_validate))
        
        if scaling == 'standardize':
            print("Standardizing data")
            mu = data_train.mean(axis=0)  # data.mean(axis=0)
            s = data_train.std(axis=0)  # data.std(axis=0)
            data_train = (data_train - mu) / s
            data_validate = (data_validate - mu) / s
            data_test = (data_test - mu) / s

        elif scaling == 'normalize':
            print("Normalizing data")
            min_val = data_train.min(axis=0)  # data.mean(axis=0)
            max_val = data_train.max(axis=0)  # data.std(axis=0)
            data_train = (data_train - min_val) / (max_val - min_val)
            data_validate = (data_validate - min_val) / (max_val - min_val)
            data_test = (data_test - min_val) / (max_val - min_val)

        return data_train, data_validate, data_test

    path = os.path.join(get_data_root(), 'data', 'miniboone/data.npy')
        
    return load_data_normalised(path, scaling)


def preprocess_and_save_miniboone(scaling='standardize'):
    train, val, test = load_miniboone(scaling)
    splits = (
        ('train', train),
        ('val', val),
        ('test', test)
    )
    for split in splits:
    
        name, data = split
        
        if scaling == 'standardize':
            file = os.path.join(get_data_root(), 'data', 'miniboone/{}.npy'.format(name))
        
        elif scaling == 'normalize':
        
            file_path = os.path.join(get_data_root(), 'data_norm', 'miniboone')
            if not os.path.exists(file_path):
                os.makedirs(file_path)
        
            file = os.path.join(file_path, '{}.npy'.format(name))
            
        print("Saving data to: " + str(file))
        np.save(file, data)


###########
# BSDS300 #
###########


def load_bsds300():
    path = os.path.join(get_data_root(), 'data', 'BSDS300/BSDS300.hdf5')
    try:
        file = h5py.File(path, 'r')
    except FileNotFoundError:
        download_data()
        file = h5py.File(path, 'r')

    return file['train'], file['validation'], file['test']


def get_data_root():
    return os.path.join(get_project_root(), 'deep_ext_obj/src/data/data_uci')


def get_pytorch_root():
    return os.path.join(get_project_root(), 'pytorch')


def get_project_root():
    return os.path.abspath('../')
