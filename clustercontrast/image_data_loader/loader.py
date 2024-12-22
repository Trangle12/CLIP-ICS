import sys
sys.path.append('..')

import torch.utils.data as data
import random


class ClassUniformlySampler(data.sampler.Sampler):
    '''
    random sample according to class label
    Arguments:
        data_source (Dataset): data_loader to sample from
        class_position (int): which one is used as class
        k (int): sample k images of each class
    '''

    def __init__(self, data_source, class_position, k):

        self.data_source = data_source
        # print(type(data_source),len(data_source))#<class 'clustercontrast.image_data_loader.dataset.PersonReIDDataSet'> 12936
        self.class_position = class_position
        self.k = k

        self.samples = self.data_source.samples
        #print("sample",self.samples)#['/data/tx/Market1501/bounding_box_train/0002_c1s1_000451_03.jpg', 0, 0, -1, 0, 0, 2016]]

        self.class_dict = self._tuple2dict(self.samples)



    def __iter__(self):
        self.sample_list = self._generate_list(self.class_dict)
        return iter(self.sample_list)

    def __len__(self):
        return len(self.sample_list)

    def _tuple2dict(self, inputs):
        '''

        :param inputs: list with tuple elemnts, [(image_path1, class_index_1), (imagespath_2, class_index_2), ...]
        :return: dict, {class_index_i: [samples_index1, samples_index2, ...]}
        '''
        dict = {}
        for index, each_input in enumerate(inputs):
            class_index = each_input[self.class_position]   # from which index to obtain the label
            if class_index not in list(dict.keys()):
                dict[class_index] = [index]
            else:
                dict[class_index].append(index)

        return dict


    def _generate_list(self, dict):
        '''
        :param dict: dict, whose values are list
        :return:
        '''

        sample_list = []

        dict_copy = dict.copy()
        keys = list(dict_copy.keys())
        random.shuffle(keys)
        for key in keys:
            value = dict_copy[key]
            if len(value) >= self.k:
                random.shuffle(value)
                sample_list.extend(value[0: self.k])
            else:
                value = value * self.k    # copy a person's image list for k-times
                random.shuffle(value)
                sample_list.extend(value[0: self.k])

        return sample_list


class IterLoader:

    def __init__(self, loader):
        self.loader = loader
        self.iter = iter(self.loader)

    # def __len__(self):
    #     return len(self.loader)
    def new_epoch(self):
        self.iter = iter(self.loader)

    def next_one(self):
        try:
            return next(self.iter)
        except:
            self.iter = iter(self.loader)
            return next(self.iter)


