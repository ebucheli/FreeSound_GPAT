from tensorflow.keras.utils import Sequence
import numpy as np

class DataGenerator(Sequence):
    def __init__(self,x_set,y_set,
                 batch_size,
                 freq_res,
                 frames):

        self.x, self.y = x_set,y_set
        self.batch_size = batch_size
        self.freq_res = freq_res
        self.frames = frames
        self.input_shape = [freq_res,frames]
        #self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.x)/self.batch_size))

    def __getitem__(self,idx):

        spects = self.x[idx*self.batch_size:(idx+1)*self.batch_size]
        batch_y = self.y[idx*self.batch_size:(idx+1)*self.batch_size]

        batch_x = self.__generate_data(spects)

        #print(batch_x.shape)
        #print(batch_y.shape)

        return batch_x,batch_y

    def __generate_data(self,spects):
        n_mels,frames = self.input_shape
        x_batch = np.zeros((len(spects),n_mels,frames))

        for i, spect in enumerate(spects):
            freq_res,time_res = spect.shape

            max_start = time_res-frames
            if max_start == 0:
                start = 0
            else:
                start = np.random.randint(0,max_start)
            end = start+frames

            x_batch[i] = spect[:,start:end]

        return x_batch

class DataGeneratorWave(Sequence):
    def __init__(self,x_set,y_set,
                 batch_size,
                 sr,file_length):

        self.x, self.y = x_set,y_set
        self.batch_size = batch_size
        self.sr = sr
        self.file_length = file_length
        #self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.x)/self.batch_size))

    def __getitem__(self,idx):

        waves = self.x[idx*self.batch_size:(idx+1)*self.batch_size]
        batch_y = self.y[idx*self.batch_size:(idx+1)*self.batch_size]

        batch_x = self.__generate_data(waves)

        #print(batch_x.shape)
        #print(batch_y.shape)

        return batch_x,batch_y

    def __generate_data(self,waves):
        #n_mels,frames = self.input_shape
        file_length = self.file_length

        x_batch = np.zeros((len(waves),file_length))

        for i, wave in enumerate(waves):
            this_length = wave.shape[0]
            #freq_res,time_res = spect.shape

            max_start = this_length-file_length
            if max_start == 0:
                start = 0
            else:
                start = np.random.randint(0,max_start)
            end = start+file_length

            x_batch[i] = wave[start:end]

        return x_batch
