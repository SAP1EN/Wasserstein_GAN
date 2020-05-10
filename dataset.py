import h5py
import torch

class celebA(torch.utils.data.Dataset):
    """
    Dataset to load images from the hdf5 file
    """
    def __init__(self, in_file, mode):
        super(celebA, self).__init__()

        self.file = h5py.File(in_file, 'r')
        total_num_imgs, self.C, self.H, self.W = self.file['images'].shape
        self.all_idxs = torch.arange(total_num_imgs)
        
        assert mode == 'train' or mode == 'test', \
            "mode should be 'train' or 'test'"
        if mode == 'train':
            self.train_mode() 
        elif mode == 'test':
            self.test_mode()

    def train_mode(self):
        """
        All except the last 9 images used for training
        """
        self.idxs = self.all_idxs[:-9] 

    def test_mode(self):
        """
        Last 9 images used for testing in Q2
        """
        self.idxs = self.all_idxs[-9:]

    def __getitem__(self, idx):
        """
        Returns images of shape C x H x W
        C = 3 (RGB)
        H = W = 64 
        """
        input = self.file['images'][self.idxs[idx],:,:,:]
        return input.astype('float32')

    def __len__(self):
        """
        self.idxs is the list of indices (into the file data)  
        that constitute the dataset
        """
        return len(self.idxs)