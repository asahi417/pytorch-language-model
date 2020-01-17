""" sequence batcher for pytorch """

import torch


class BatchFeeder:
    """ Pytorch batch feeding iterator for language model training """

    def __init__(self,
                 batch_size,
                 num_steps,
                 sequence,
                 cuda: bool=False):
        """ Pytorch batch feeding iterator for language model training

         Parameter
        -------------------
        batch_size: int
            batch size
        num_steps: int
            sequence truncation size
        sequence: list
            integer token id sequence to feed
        """
        self._index = 0
        self.batch_size = batch_size
        self.num_steps = num_steps
        seq = torch.LongTensor(sequence)
        self.data_size = seq.size(0)

        n_batch = self.data_size // self.batch_size
        # Trim off any extra elements that wouldn't cleanly fit (remainders).
        seq = seq.narrow(0, 0, n_batch * self.batch_size)
        # Evenly divide the data across the bsz batches.
        self._data = seq.view(self.batch_size, -1).t().contiguous()
        print(self._data.shape)
        if cuda and torch.cuda.device_count() >= 1:
            self._data.cuda()

    def __len__(self):
        return (self.data_size // self.batch_size - 1) // self.num_steps

    def __iter__(self):
        return self

    def __next__(self):
        """ next batch for train data (size is `self._batch_size`) loop for self._iteration_number

         Return
        -----------------
        (inputs, outputs): list (batch_size, num_steps)
        """
        if (self._index + 1) * self.num_steps + 1 > self._data.size(0):
            self._index = 0
            raise StopIteration
        x = self._data[self._index * self.num_steps:(self._index + 1) * self.num_steps, :]
        y = self._data[self._index * self.num_steps + 1:(self._index + 1) * self.num_steps + 1, :]
        self._index += 1
        return x, y


if __name__ == '__main__':
    # with open('./data/penn-treebank/ptb.test.eos.id.txt', 'r') as f:
    #     text = [int(i) for i in f.read().split()]
    # bf = BatchFeeder(batch_size=5, num_steps=6, sequence=[i for i in text])
    bf = BatchFeeder(batch_size=5, num_steps=6, sequence=[i for i in range(100)])
    print(len(bf))
    for n, i in enumerate(bf):
        _x, _y = i
        print(_x)
        print(_y)
        print()
        if n > 10:
            break

