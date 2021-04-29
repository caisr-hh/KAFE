"""
@author: awaash
"""

import numpy as np
import torch
from torch.utils.data import Dataset

class DataReader:

    def __init__(self, inputFileName, min_count, root_size, adv_thresh):
        self.negatives = []
        self.discards = []
        self.negpos = 0
        self.root_size = root_size
        self.adv_thresh = adv_thresh
        self.word2id = dict()
        self.id2word = dict()
        self.sentences_count = 0
        self.token_count = 0
        self.word_frequency = dict()
        self.word_label = dict()
        self.root_vectors = None
        self.root2id = dict()
        self.id2root = dict()
        self.root_map = []
        self.inputFileName = inputFileName
        self.read_words(min_count)
        self.initTableNegatives()
        self.initTableDiscards()

    def read_words(self, min_count):
        word_frequency = dict()
        word_frequency_new = dict()
        for line in open(self.inputFileName, encoding="utf8"):
            line = line.split()
            if len(line) > 1:
                self.sentences_count += 1
                for word in line:
                    if len(word) > 0:
                        self.token_count += 1
                        word_frequency[word] = word_frequency.get(word, 0) + 1
                        word_frequency_new[word] = word_frequency_new.get(word, 0) + (len(line)) - 1

                        if self.token_count % 1000000 == 0:
                            print("Read " + str(int(self.token_count / 1000000)) + "M words.")

        wid = 0
        rid = 0
        for w, c in word_frequency.items():
            if c < min_count:
                continue
            self.word2id[w] = wid
            self.id2word[wid] = w
            self.word_frequency[wid] = c
            if w[0:self.root_size] not in self.root2id:
                self.root2id[w[0:self.root_size]] = rid
                self.id2root[rid] = w[0:self.root_size]
                rid +=1
            wid += 1

        self.root_map = [self.root2id[k[0:self.root_size]] for k in self.word2id.keys()]


        adv_bias = np.percentile(list(self.word_frequency.values()), self.adv_thresh)
        for w, c in self.word_frequency.items():
            if c < adv_bias:
                self.word_label[w] = 1
            else:
                self.word_label[w] = 0

        print("Total embeddings: " + str(len(self.word2id)))
        print("Adversarial bias is", str(adv_bias))

    def initTableDiscards(self):
        t = 0.01
        f = np.array(list(self.word_frequency.values())) / self.token_count
        self.discards = np.sqrt(t / f) + (t / f)

    def initTableNegatives(self):
        NEGATIVE_TABLE_SIZE = 1e6
        pow_frequency = np.array(list(self.word_frequency.values())) ** 0.5
        words_pow = sum(pow_frequency)
        ratio = pow_frequency / words_pow  
        print(min(ratio), max(ratio))          
        print(ratio)
        count = np.round(ratio * NEGATIVE_TABLE_SIZE)
        for wid, c in enumerate(count):
            self.negatives += [wid] * int(c)
        self.negatives = np.array(self.negatives)
        np.random.shuffle(self.negatives)

    def getNegatives(self,u_in, v_in, size):
        target = u_in + v_in
        response = self.negatives[self.negpos:self.negpos + size]
        while target in response:
            response = self.negatives[self.negpos:self.negpos + size]
            self.negpos = (self.negpos + size) % len(self.negatives)
            if len(response) != size:
                response = np.concatenate((response, self.negatives[0:self.negpos]))
        self.negpos = (self.negpos + size) % len(self.negatives)
        if len(response) != size:
            return np.concatenate((response, self.negatives[0:self.negpos]))
        return response

    def getRoots(self, i):
        return self.root2id[self.id2word[i][0:self.root_size]]

    def getLabels(self, i):
        return self.word_label[i]


# -----------------------------------------------------------------------------------------------------------------

class KAFEdataset(Dataset):
    
    def __init__(self, data, window_size, neg_samples):
        self.data = data
        self.window_size = window_size
        self.neg_samples = neg_samples
        self.input_file = open(data.inputFileName, encoding="utf8")

    def __len__(self):
        return self.data.sentences_count

    def __getitem__(self, idx):
        while True:
            line = self.input_file.readline()
            if not line:
                self.input_file.seek(0, 0)
                line = self.input_file.readline()

            if len(line) > 1:
                words = line.split()

                if len(words) > 1:
                    word_ids = [self.data.word2id[w] for w in words if
                                w in self.data.word2id and np.random.rand() < self.data.discards[self.data.word2id[w]]]
                    boundary = np.random.randint(1, self.window_size)
                    return [(u, v, self.data.getNegatives(v,u, self.neg_samples), self.data.getRoots(u), self.data.getLabels(u))
                            for i, u in enumerate(word_ids) for j, v in
                            enumerate(word_ids[max(i - boundary, 0):i + boundary]) if u != v]

    @staticmethod
    def collate(batches):
        all_u = [u for batch in batches for u, _, _, _, _ in batch if len(batch) > 0]
        all_v = [v for batch in batches for _, v, _, _, _ in batch if len(batch) > 0]
        all_neg_v = [neg_v for batch in batches for _, _, neg_v, _, _ in batch if len(batch) > 0]
        all_r = [r for batch in batches for _, _, _, r, _ in batch if len(batch) > 0]
        all_lbl = [l for batch in batches for _, _, _, _, l in batch if len(batch) > 0]

        return torch.LongTensor(all_u), torch.LongTensor(all_v), torch.LongTensor(all_neg_v), \
               torch.LongTensor(all_r), torch.FloatTensor(all_lbl)
