import numpy as np
from PIL import Image
import copy
import os
import scipy.io as sio
import torch

def os_walk(folder_dir):

    for root, dirs, files in os.walk(folder_dir):
        files = sorted(files, reverse=False)
        dirs = sorted(dirs, reverse=False)
        return root, dirs, files


class PersonReIDSamples:

    def __init__(self, samples_path, reorder=True, get_semi_label=True, save_semi_gt_ID=False):

        # parameters
        self.samples_path = samples_path
        self.reorder = reorder
        self.get_semi_label = get_semi_label
        self.save_semi_gt_ID = save_semi_gt_ID

        # load samples
        samples = self._load_images_path(self.samples_path)

        # reorder person identities and camera identities
        if self.reorder:
            samples = self._reorder_labels(samples, 1)  # re-order so that labels are 0,1,2,3,...
            samples = self._reorder_labels(samples, 2)  # re-order so that cams are 0,1,2,3,...

        if self.get_semi_label:
            samples, id_count_each_cam, img_count_each_cam, semi_label_each_cam = self._extend_semi_labels(samples)
            self.id_count_each_cam = id_count_each_cam
            self.img_count_each_cam = img_count_each_cam
            self.semi_label_each_cam = semi_label_each_cam
        self.samples = samples


    def _reorder_labels(self, samples, label_index):

        ids = []
        for sample in samples:
            ids.append(sample[label_index])

        # delete repetitive elments and order
        ids = list(set(ids))
        ids.sort()
        # reorder
        for sample in samples:
            sample[label_index] = ids.index(sample[label_index])

        return samples


    def _extend_semi_labels(self, samples):
        ids, cams = [], []
        for sample in samples:
            ids.append(sample[1])
            cams.append(sample[2])
        ids = np.array(ids)
        cams = np.array(cams)

        semi_labels = np.zeros(ids.shape, ids.dtype)
        accumulate_labels = np.zeros(ids.shape, ids.dtype)
        img_idx_labels = np.zeros(ids.shape, ids.dtype)  # index of each image, seperately for each camera 
   
        id_count_each_cam = []
        img_count_each_cam = []
        semi_label_each_cam = []
        unique_cams = np.unique(cams)
        prev_id_count = 0
        print('  unique cameras= {}'.format(unique_cams))

        for this_cam in unique_cams:
            percam_labels = ids[cams == this_cam]
            img_count_each_cam.append(len(percam_labels))
            unique_id = np.unique(percam_labels)
            id_count_each_cam.append(len(unique_id))
            id_dict = {ID: i for i, ID in enumerate(unique_id.tolist())}

            for i in range(len(percam_labels)):
                percam_labels[i] = id_dict[percam_labels[i]]
            
            semi_labels[cams == this_cam] = percam_labels
            accumulate_labels[cams == this_cam] = percam_labels + prev_id_count
            img_idx_labels[cams == this_cam] = np.arange(len(percam_labels))
            semi_label_each_cam.append(percam_labels)
            prev_id_count += len(unique_id)
        assert prev_id_count == (np.max(accumulate_labels) + 1)

        # extend labels
        for i, sample in enumerate(samples):
            sample.append(semi_labels[i])
            sample.append(accumulate_labels[i])
            sample.append(img_idx_labels[i])

        return samples, id_count_each_cam, img_count_each_cam, semi_label_each_cam


    def _load_images_path(self, folder_dir):
        '''
        :param folder_dir:
        :return: [ [path, identiti_id, camera_id], ]
        '''
        samples = []


        root_path, _, files_name = os_walk(folder_dir)  # images under the folder are read in random order
        all_img_ids, all_img_cams = [], []
        for file_name in files_name:
            if '.jpg' in file_name:
                identi_id, camera_id = self._analysis_file_name(file_name)
                all_img_ids.append(identi_id)
                all_img_cams.append(camera_id)
                samples.append([root_path + file_name, identi_id, camera_id, -1])  # -1 is the target camera style

        all_img_ids = np.expand_dims(all_img_ids, axis=1)
        all_img_cams = np.expand_dims(all_img_cams, axis=1)
        all_img_ids_cams = np.concatenate((all_img_ids, all_img_cams), axis=1)

        return samples

    def _analysis_file_name(self, file_name):
        '''

        :param file_name: format like 0844_c3s2_107328_01.jpg
        :return:
        '''
        split_list = file_name.replace('.jpg', '').replace('c', '').replace('s', '_').split('_')
        identi_id, camera_id = int(split_list[0]), int(split_list[1])
        return identi_id, camera_id



class Samples4Market(PersonReIDSamples):
    '''Market Dataset
    '''
    pass



class Samples4MarketAugmented(PersonReIDSamples):
    '''augmented Market Dataset
    '''

    def _load_images_path(self, folder_dir):
        '''
        :param folder_dir:
        :return: [ [path, identiti_id, camera_id], ]
        '''
        samples = []
        root_path, _, files_name = os_walk(folder_dir)
        for file_name in files_name:
            if '.jpg' in file_name:
                identi_id, camera_id, Tcam_id = self._analysis_file_name(file_name)
                samples.append([root_path + file_name, identi_id, camera_id, Tcam_id])  # Tcam_id is the target camera style

        samples = self._reorder_labels(samples, 3)  # re-order Tcam label as 0,1,2,3,...
        return samples


    def _analysis_file_name(self, file_name):
        '''
        :param file_name: format like 0589_c3s2_006893_03_623.jpg
        :return:
        '''
        split_list = file_name.replace('.jpg', '').split('_')
        identi_id, camera_id, Tcam_id = int(split_list[0]), int(split_list[-1][0]), int(split_list[-1][2])  # camera_id: original camera id
        return identi_id, camera_id, Tcam_id



class Samples4Duke(PersonReIDSamples):
    '''Duke dataset
    '''
    def _analysis_file_name(self, file_name):
        '''
        :param file_name: format like 0002_c1_f0044158.jpg
        :return:
        '''
        split_list = file_name.replace('.jpg', '').replace('c', '').split('_')
        identi_id, camera_id = int(split_list[0]), int(split_list[1])
        return identi_id, camera_id


class PersonReIDDataSet:

    def __init__(self, samples, transform=None, aug_transform=None, mutual=False):
        #print("samples",type(samples),len(samples))#samples <class 'list'> 12936
        self.samples = samples
        self.transform = transform
        self.aug_transform = aug_transform
        self.mutual = mutual

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        if self.mutual:
            return self._get_mutual_item(index)
        else:
            return self._get_single_item(index)

    def _get_mutual_item(self, index):

        this_sample = copy.deepcopy(self.samples[index])

        this_sample[0] = self._loader_mutual(this_sample[0])

        for m in range(1, len(this_sample)):
            this_sample[m] = np.array(this_sample[m])  # ID, cam, Tcam, semi_label, accum_label, predicted_label

        return this_sample

    def _get_single_item(self, index):

        this_sample = copy.deepcopy(self.samples[index])

        this_sample[0] = self._loader_single(this_sample[0])

        for m in range(1, len(this_sample)):
            this_sample[m] = np.array(this_sample[m])  # ID, cam, Tcam, semi_label, accum_label, predicted_label

        return this_sample

    def _loader_single(self, img_path):
        img_1 = Image.open(img_path).convert('RGB')

        if self.transform is not None:
            img_1 = self.transform(img_1)

        return img_1

    def _loader_mutual(self, img_path):
        img = Image.open(img_path).convert('RGB')

        if self.transform is not None:
            img1 = self.transform(img)
        if self.aug_transform is not None:
            img2 = self.aug_transform(img)

        return (img1 , img2)

class PersonReIDDataSetAugment:

    def __init__(self, samples, transform, num_cam=6):
        self.samples = samples
        self.transform = transform
        self.num_cam = num_cam

        cams, semi_labels = [], []
        for ii in range(100):
            cams.append(self.samples[ii][2])
            semi_labels.append(self.samples[ii][4])

    def __getitem__(self, index):

        samples = []

        # ----------------- original img ------------------
        this_sample = copy.deepcopy(self.samples[index])
        ori_img_name = this_sample[0]
        this_sample[0] = self._loader(this_sample[0])
        if self.transform is not None:
            this_sample[0] = self.transform(this_sample[0])

        this_sample.append(index)  # a unique index to mark each individual image
        for m in range(1, len(this_sample)):
            this_sample[m] = np.array(this_sample[m])  # ID, cam, Tcam, semi_label, accum_label, (predicted_label), img_idx

        samples.append(this_sample)

        # -------------- transferred counterpart --------------
        #     '/home1/wml/dataset/Market1501/stargan_augmented/bounding_box_train/0589_c1s3_035001_03.jpg'
        # '/home1/wml/dataset/Market1501/stargan_augmented/bounding_box_train_aug/0589_c4s3_035001_03_124.jpg'
        ori_root = ori_img_name[:-24]  # '/home1/wml/dataset/Market1501/stargan_augmented/bounding_box_train'

        split_list = ori_img_name[-23:].replace('.jpg', '').split('c')
        ori_cam = int(split_list[1][0])

        # randomly get a different cam to transfer to
        sel_cam = ori_cam
        while sel_cam == ori_cam:
            sel_cam = np.random.permutation(self.num_cam)[0] + 1

        transfer_path = ori_root + '_stargan_aug/' + split_list[0] + 'c' + str(sel_cam) + split_list[1][1:] + '_' + str(ori_cam) + '2' + str(sel_cam) + '.jpg'
        transfer_sample = copy.deepcopy(self.samples[index])
        transfer_sample[0] = self._loader(transfer_path)
        if self.transform is not None:
            transfer_sample[0][0] = self.transform(transfer_sample[0])
            transfer_sample[0][1] = self.transform(transfer_sample[0])

        transfer_sample.append(index)  # a unique index to mark each individual image, AND its transferred counterpart
        for m in range(1, len(transfer_sample)):
            if m == 3:
                transfer_sample[m] = sel_cam
            transfer_sample[m] = np.array(transfer_sample[m])  # ID, cam, Tcam, semi_label, accum_label, (predicted_label), img_idx

        samples.append(transfer_sample)

        return samples

    def __len__(self):
        return len(self.samples)

    def _loader(self, img_path):
        return Image.open(img_path).convert('RGB')



if __name__ == '__main__':
    import torchvision.transforms as transforms
    from clustercontrast.utils.data import transforms as T
    samples = PersonReIDSamples(r'E:\Project\need_data\Market-1501-v15.09.15\bounding_box_train').samples
    transform_train = transforms.Compose([
        transforms.Resize([256,128], interpolation=3),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.Pad(10),
        transforms.RandomCrop([256,128]),
        # transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        T.RandomErasing(probability=0.5, mean=[0.485, 0.456, 0.406])
        ])
    dataset = PersonReIDDataSet(samples,transform=transform_train)
    print(dataset.__len__(),dataset.__getitem__(index=3),type(dataset))

