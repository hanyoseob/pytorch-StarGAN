import numpy as np
import torch
from skimage import transform
import matplotlib.pyplot as plt
import os


class Dataset(torch.utils.data.Dataset):
    """
    dataset of image files of the form 
       stuff<number>_trans.pt
       stuff<number>_density.pt
    """

    def __init__(self, data_dir, attrs=[], data_type='float32', transform=[], mode='train'):
        self.data_dir = data_dir
        self.transform = transform
        self.data_type = data_type
        self.attrs = attrs
        self.mode = mode

        data_name = data_dir.split('/')[-1]

        if data_name == 'celeba':
            lines = [line.rstrip() for line in open(os.path.join(self.data_dir, 'list_attr_celeba.txt'), 'r')]
            all_attr_names = lines[1].split()
            lines = lines[2:]

        elif data_name == 'rafd':
            lines = os.listdir(self.data_dir)
            all_attr_names = ['angry', 'contemptuous', 'disgusted', 'fearful', 'happy', 'neutral', 'sad', 'surprised']

        np.random.seed(1234)
        np.random.shuffle(lines)

        attr2idx = {}
        idx2attr = {}
        for i, attr_name in enumerate(all_attr_names):
            attr2idx[attr_name] = i
            idx2attr[i] = attr_name

        self.attr2idx = attr2idx
        self.idx2attr = idx2attr

        train_dataset = []
        test_dataset = []

        if data_name == 'celeba':
            for i, line in enumerate(lines):
                split = line.split()
                filename = split[0]
                values = split[1:]

                label = []
                for attr_name in self.attrs:
                    idx = self.attr2idx[attr_name]
                    label.append(values[idx] == '1')

                if (i + 1) > 2000:
                    train_dataset.append([filename, label])
                else:
                    test_dataset.append([filename, label])

        elif data_name == 'rafd':
            for i, line in enumerate(lines):
                label = np.zeros(len(self.attrs), dtype=np.float32)
                split = line.split('_')
                filename = line
                attr = split[4]

                label[attr2idx[attr]] = 1

                if (i + 1) <= 4000:
                    train_dataset.append([filename, label])
                else:
                    test_dataset.append([filename, label])

        if self.mode == 'train':
            self.dataset = train_dataset
        else:
            self.dataset = test_dataset

    def __getitem__(self, index):

        # x = np.load(os.path.join(self.data_dir, self.names[0][index]))
        # y = np.load(os.path.join(self.data_dir, self.names[1][index]))
        filename, label = self.dataset[index]
        data = plt.imread(os.path.join(self.data_dir, filename)).squeeze()

        if data.dtype == np.uint8:
            data = data / 255.0

        if len(data.shape) == 2:
            data = np.expand_dims(data, axis=2)
            data = np.tile(data, (1, 1, 3))

        if self.transform:
            data = self.transform(data)
            label = torch.FloatTensor(label)

        return data, label

    def __len__(self):
        return len(self.dataset)


class ToTensor(object):
    def __call__(self, data):
        data = data.transpose((2, 0, 1)).astype(np.float32)
        return torch.from_numpy(data)


class Normalize(object):
    def __call__(self, data):
        data = 2 * data - 1
        return data


class RandomFlip(object):
    def __call__(self, data):
        if np.random.rand() > 0.5:
            data = np.fliplr(data)

        return data


class Rescale(object):
    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, data):
        h, w = data.shape[:2]

        if isinstance(self.output_size, int):
          if h > w:
            new_h, new_w = self.output_size * h / w, self.output_size
          else:
            new_h, new_w = self.output_size, self.output_size * w / h
        else:
          new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        data = transform.resize(data, (new_h, new_w))
        return data


class CenterCrop(object):
    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, data):
        h, w = data.shape[:2]

        new_h, new_w = self.output_size

        top = int(abs(h - new_h) / 2)
        left = int(abs(w - new_w) / 2)

        data = data[top: top + new_h, left: left + new_w]

        return data


class RandomCrop(object):

    def __init__(self, output_size):

        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, data):
        h, w = data.shape[:2]

        new_h, new_w = self.output_size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        data = data[top: top + new_h, left: left + new_w]
        return data


class ToNumpy(object):
    def __call__(self, data):

        if data.ndim == 3:
            data = data.to('cpu').detach().numpy().transpose((1, 2, 0))
        elif data.ndim == 4:
            data = data.to('cpu').detach().numpy().transpose((0, 2, 3, 1))

        return data


class Denomalize(object):
    def __call__(self, data):

        return (data + 1) / 2
