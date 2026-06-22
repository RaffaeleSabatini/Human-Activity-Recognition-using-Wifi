import numpy as np
import numpy.random as rnd

from os import listdir, makedirs, path
from glob import glob

def create_test_dataset(dataset_path, doppler_trace_size):
    test_dataset  = dataset_path.split('/')[0] + "_test"

    if test_dataset not in listdir():
        print(f"Creating {test_dataset}")
        makedirs(test_dataset, exist_ok=True)

        subdir_list = sorted([name for name in glob(dataset_path) if path.isdir(name)])
        for subdir in subdir_list:
            traces_list = sorted([trace for trace in listdir(subdir) if trace.endswith(".txt")])

            for trace_name in traces_list:
                stream     = int(trace_name[-5])

                full_trace = np.load(subdir+'/'+trace_name, allow_pickle=True)
                N_images   = full_trace.shape[0]//doppler_trace_size

                for i in range(N_images):
                    image_idx = i*4+stream

                    np.save(test_dataset+'/'+trace_name[:-5]+str(image_idx), full_trace[i*doppler_trace_size:(i+1)*doppler_trace_size, :], allow_pickle=True)
        print("Train and test datasets created successfully!")
    else:
        print("Dataset is already splitted!")