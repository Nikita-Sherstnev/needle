import numpy as np
from .autograd import Tensor
import os
import gzip
import pickle
from typing import Iterator, Optional, List, Sized, Union, Iterable, Any
from needle import backend_ndarray as nd


class Transform:
    def __call__(self, x):
        raise NotImplementedError


class RandomFlipHorizontal(Transform):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img):
        """
        Horizonally flip an image, specified as n H x W x C NDArray.
        Args:
            img: H x W x C NDArray of an image
        Returns:
            H x W x C ndarray corresponding to image flipped with probability self.p
        Note: use the provided code to provide randomness, for easier testing
        """
        flip_img = np.random.rand() < self.p
        ### BEGIN YOUR SOLUTION
        if flip_img:
            img = np.flip(img, 1)
        return img
        ### END YOUR SOLUTION


class RandomCrop(Transform):
    def __init__(self, padding=3):
        self.padding = padding

    def __call__(self, img):
        """Zero pad and then randomly crop an image.
        Args:
             img: H x W x C NDArray of an image
        Return
            H x W x C NAArray of cliped image
        Note: generate the image shifted by shift_x, shift_y specified below
        """
        shift_x, shift_y = np.random.randint(
            low=-self.padding, high=self.padding + 1, size=2
        )
        ### BEGIN YOUR SOLUTION
        p = self.padding
        shift_x, shift_y = shift_x+p, shift_y+p
        res = np.pad(img, ((p, p), (p, p), (0, 0)))[shift_x:img.shape[0]+shift_x,shift_y:img.shape[1]+shift_y,:]
        return res
        ### END YOUR SOLUTION


class Dataset:
    r"""An abstract class representing a `Dataset`.

    All subclasses should overwrite :meth:`__getitem__`, supporting fetching a
    data sample for a given key. Subclasses must also overwrite
    :meth:`__len__`, which is expected to return the size of the dataset.
    """

    def __init__(self, transforms: Optional[List] = None):
        self.transforms = transforms

    def __getitem__(self, index) -> object:
        raise NotImplementedError

    def __len__(self) -> int:
        raise NotImplementedError

    def apply_transforms(self, x):
        if self.transforms is not None:
            # apply the transforms
            for tform in self.transforms:
                x = tform(x)
        return x


class DataLoader:
    r"""
    Data loader. Combines a dataset and a sampler, and provides an iterable over
    the given dataset.
    Args:
        dataset (Dataset): dataset from which to load the data.
        batch_size (int, optional): how many samples per batch to load
            (default: ``1``).
        shuffle (bool, optional): set to ``True`` to have the data reshuffled
            at every epoch (default: ``False``).
    """
    dataset: Dataset
    batch_size: Optional[int]

    def __init__(
        self,
        dataset: Dataset,
        batch_size: Optional[int] = 1,
        shuffle: bool = False,
    ):

        self.dataset = dataset
        self.shuffle = shuffle
        self.batch_size = batch_size
        if not self.shuffle:
            self.ordering = np.array_split(
                np.arange(len(dataset)), range(batch_size, len(dataset), batch_size)
            )

    def __iter__(self):
        ### BEGIN YOUR SOLUTION
        self.index = 0
        if self.shuffle:
            indices = np.arange(len(self.dataset))
            np.random.shuffle(indices)
            self.ordering = np.array_split(indices,
                                           range(self.batch_size, len(self.dataset), self.batch_size))

        if len(self.ordering) > 1:
            if len(self.ordering[-1]) < len(self.ordering[-2]):
                del self.ordering[-1]
        ### END YOUR SOLUTION
        return self

    def __next__(self):
        ### BEGIN YOUR SOLUTION
        if self.index >= len(self.ordering):
            raise StopIteration

        data = []

        if len(self.dataset[0]) == 2:
            for el in self.ordering[self.index]:
                if isinstance(self.dataset[el], tuple):
                    for inst in zip([self.dataset[el][0]], [self.dataset[el][1]]):
                        data.append(inst)
                else:
                    for inst in zip(self.dataset[el][0], self.dataset[el][1]):
                        data.append(inst)

                imgs = []
                labels = []
                for inst in data:
                    imgs.append(inst[0])
                    labels.append(inst[1])

            data = (np.array(imgs), np.array(labels))
            data = [Tensor(d) for d in data]
        else:
            for el in self.ordering[self.index]:
                data.append(self.dataset[el])

            data = np.concatenate(data, axis=0)
            data = [Tensor(data)]

        self.index += 1

        return data
        ### END YOUR SOLUTION


class MNISTDataset(Dataset):
    def __init__(
        self,
        image_filename: str,
        label_filename: str,
        transforms: Optional[List] = None,
    ):
        ### BEGIN YOUR SOLUTION
        with gzip.open(image_filename, 'r') as f:
            magic_number = int.from_bytes(f.read(4), 'big')
            image_count = int.from_bytes(f.read(4), 'big')
            row_count = int.from_bytes(f.read(4), 'big')
            column_count = int.from_bytes(f.read(4), 'big')

            image_data = f.read()
            images = np.frombuffer(image_data, dtype=np.uint8)\
                        .reshape((image_count, row_count * column_count))
            images = images.astype(np.float32)
            images = images / 255.0

        with gzip.open(label_filename, 'r') as f:
            magic_number = int.from_bytes(f.read(4), 'big')
            label_count = int.from_bytes(f.read(4), 'big')
            label_data = f.read()
            labels = np.frombuffer(label_data, dtype=np.uint8)

        self.images = images
        self.labels = labels
        self.transforms = transforms if transforms else []
        ### END YOUR SOLUTION

    def __getitem__(self, index) -> object:
        ### BEGIN YOUR SOLUTION
        res = []
        imgs = []
        labels = []
        if isinstance(index, slice):
            for ind in range(index.start, index.stop):
                img = self.images[ind].reshape(28, 28, 1)
                img = self.apply_transforms(img).flatten()
                imgs.append(img)
                labels.append(self.labels[ind])
        else:
            img = self.images[index].reshape(28, 28, 1)
            img = self.apply_transforms(img).flatten()
            return (img, self.labels[index])

        return (np.array(imgs), np.array(labels))
        ### END YOUR SOLUTION

    def __len__(self) -> int:
        ### BEGIN YOUR SOLUTION
        return self.images.shape[0]
        ### END YOUR SOLUTION


class CIFAR10Dataset(Dataset):
    def __init__(
        self,
        base_folder: str,
        train: bool,
        p: Optional[int] = 0.5,
        transforms: Optional[List] = None
    ):
        """
        Parameters:
        base_folder - cifar-10-batches-py folder filepath
        train - bool, if True load training dataset, else load test dataset
        Divide pixel values by 255. so that images are in 0-1 range.
        Attributes:
        X - numpy array of images
        y - numpy array of labels
        """
        ### BEGIN YOUR SOLUTION

        imgs = []
        labels = []
        files = os.listdir(base_folder)
        for f in files:
            if train and f.startswith("data") or not train and f.startswith("test"):
                with open(os.path.join(base_folder, f), 'rb') as fo:
                    data = pickle.load(fo, encoding='bytes')
                    imgs.append(data[b'data'])
                    labels.append(data[b'labels'])
        self.X = np.concatenate(imgs).astype('float32').reshape((-1,3,32,32)) / 255.
        self.y = np.concatenate(labels).astype('float32')
        self.transforms = transforms if transforms else []
        ### END YOUR SOLUTION

    def __getitem__(self, index) -> object:
        """
        Returns the image, label at given index
        Image should be of shape (3, 32, 32)
        """
        ### BEGIN YOUR SOLUTION
        # res = []
        # imgs = []
        # labels = []
        # if isinstance(index, slice):
        #     for ind in range(index.start, index.stop):
        #         img = self.X[ind].reshape(3,32,32)
        #         img = self.apply_transforms(img)
        #         imgs.append(img)
        #         labels.append(self.y[ind])
        # else:
        img = self.X[index]
        img = self.apply_transforms(img)
        return (img, self.y[index])

        #return np.array(imgs), np.array(labels)
        ### END YOUR SOLUTION

    def __len__(self) -> int:
        """
        Returns the total number of examples in the dataset
        """
        ### BEGIN YOUR SOLUTION
        return self.X.shape[0]
        ### END YOUR SOLUTION


class NDArrayDataset(Dataset):
    def __init__(self, *arrays):
        self.arrays = arrays

    def __len__(self) -> int:
        return self.arrays[0].shape[0]

    def __getitem__(self, i) -> object:
        return tuple([a[i] for a in self.arrays])




class Dictionary(object):
    """
    Creates a dictionary from a list of words, mapping each word to a
    unique integer.
    Attributes:
    word2idx: dictionary mapping from a word to its unique ID
    idx2word: list of words in the dictionary, in the order they were added
        to the dictionary (i.e. each word only appears once in this list)
    """
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []

    def find(self, word):
        try:
            idx = self.word2idx[word]
        except KeyError:
            idx = None
        return idx

    def lookup(self, idx):
        if idx >= 0 and idx < len(self):
            return self.idx2word[idx]
        else:
            return None

    def add_word(self, word):
        """
        Input: word of type str
        If the word is not in the dictionary, adds the word to the dictionary
        and appends to the list of words.
        Returns the word's unique ID.
        """
        ### BEGIN YOUR SOLUTION
        idx = self.find(word)
        if idx is None:
            idx = len(self)
            self.word2idx[word] = idx
            self.idx2word.append(word)
        return idx
        ### END YOUR SOLUTION

    def __len__(self):
        """
        Returns the number of unique words in the dictionary.
        """
        ### BEGIN YOUR SOLUTION
        return len(self.idx2word)
        ### END YOUR SOLUTION



class Corpus(object):
    """
    Creates corpus from train, and test txt files.
    """
    def __init__(self, base_dir, max_lines=None):
        self.dictionary = Dictionary()
        self.train = self.tokenize(os.path.join(base_dir, 'train.txt'), max_lines)
        self.test = self.tokenize(os.path.join(base_dir, 'test.txt'), max_lines)

    def tokenize(self, path, max_lines=None):
        """
        Input:
        path - path to text file
        max_lines - maximum number of lines to read in
        Tokenizes a text file, first adding each word in the file to the dictionary,
        and then tokenizing the text file to a list of IDs. When adding words to the
        dictionary (and tokenizing the file content) '<eos>' should be appended to
        the end of each line in order to properly account for the end of the sentence.
        Output:
        ids: List of ids
        """
        ### BEGIN YOUR SOLUTION
        lines = open(path).readlines()
        if max_lines is None:
            max_lines = len(lines)
        ids = []
        for line in lines[:max_lines]:
            words = line.split() + ["<eos>"]
            ids += [self.dictionary.add_word(w) for w in words]
        return ids
        ### END YOUR SOLUTION

    def lookup(self, idx):
        return self.dictionary.lookup(idx)


def batchify(data, batch_size, device, dtype):
    """
    Starting from sequential data, batchify arranges the dataset into columns.
    For instance, with the alphabet as the sequence and batch size 4, we'd get
    ┌ a g m s ┐
    │ b h n t │
    │ c i o u │
    │ d j p v │
    │ e k q w │
    └ f l r x ┘.
    These columns are treated as independent by the model, which means that the
    dependence of e. g. 'g' on 'f' cannot be learned, but allows more efficient
    batch processing.
    If the data cannot be evenly divided by the batch size, trim off the remainder.
    Returns the data as a numpy array of shape (nbatch, batch_size).
    """
    ### BEGIN YOUR SOLUTION
    nbatch = len(data) // batch_size
    x = np.array(data[0 : nbatch * batch_size]).astype(dtype)
    return x.reshape((batch_size, nbatch)).transpose()
    ### END YOUR SOLUTION


def get_batch(batches, i, bptt, device=None, dtype=None):
    """
    get_batch subdivides the source data into chunks of length bptt.
    If source is equal to the example output of the batchify function, with
    a bptt-limit of 2, we'd get the following two Variables for i = 0:
    ┌ a g m s ┐ ┌ b h n t ┐
    └ b h n t ┘ └ c i o u ┘
    Note that despite the name of the function, the subdivison of data is not
    done along the batch dimension (i.e. dimension 1), since that was handled
    by the batchify function. The chunks are along dimension 0, corresponding
    to the seq_len dimension in the LSTM or RNN.
    Inputs:
    batches - numpy array returned from batchify function
    i - index
    bptt - Sequence length
    Returns:
    data - Tensor of shape (bptt, bs) with cached data as NDArray
    target - Tensor of shape (bptt*bs,) with cached data as NDArray
    """
    ### BEGIN YOUR SOLUTION
    n = batches.shape[0]

    seq_len = min(bptt, n - 1 - i)
    data   = batches[i:i+seq_len,]
    target = batches[i+1:i+1+seq_len,].flatten()

    return Tensor(data, device=device, dtype=dtype), Tensor(target, device=device, dtype=dtype)
    ### END YOUR SOLUTION