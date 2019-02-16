import os
import torch
import pdb

class Dictionary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)


class Corpus(object):
    def __init__(self, path):
        self.dictionary = Dictionary()
        # modified to accomendate multiple files
        self.train_path = os.path.join(path, 'train/')
        self.valid_path = os.path.join(path, 'validation/')
        self.test_path = os.path.join(path, 'test/')

        self.train_files = [os.path.join(self.train_path, f) for f in os.listdir(self.train_path) if os.path.isfile(os.path.join(self.train_path, f)) and f != '.DS_Store' ]
        self.valid_files = [os.path.join(self.valid_path, f) for f in os.listdir(self.valid_path) if os.path.isfile(os.path.join(self.valid_path, f)) and f != '.DS_Store' ]
        self.test_files = [os.path.join(self.test_path, f) for f in os.listdir(self.test_path) if os.path.isfile(os.path.join(self.test_path, f)) and f != '.DS_Store' ]

        self.train_tokens = 0
        self.valid_tokens = 0
        self.test_tokens = 0

        self.train_tokens_index = 0
        self.valid_tokens_index = 0
        self.test_tokens_index = 0

        # populate dictionary
        for file in self.train_files:
            self.train_tokens += self.populate_dictionary(file)

        for file in self.valid_files:
            self.valid_tokens += self.populate_dictionary(file)

        for file in self.test_files:
            self.test_tokens += self.populate_dictionary(file)

        # tokenize the files into a tensor
        self.train = torch.LongTensor(self.train_tokens)
        self.valid = torch.LongTensor(self.valid_tokens)
        self.test = torch.LongTensor(self.test_tokens)

        for file in self.train_files:
            self.train_tokens_index += self.tokenize(self.train, self.train_tokens_index, file)

        for file in self.valid_files:
            self.valid_tokens_index += self.tokenize(self.valid, self.valid_tokens_index, file)

        for file in self.test_files:
            self.test_tokens_index += self.tokenize(self.test, self.test_tokens_index, file)

    def populate_dictionary(self, path):
        """Tokenizes a text file."""
        assert os.path.exists(path)
        # Add words to the dictionary
        # pdb.set_trace()
        with open(path, 'r') as f:
            tokens = 0
            for line in f:
                words = line.split() + ['<eos>']
                tokens += len(words)
                for word in words:
                    self.dictionary.add_word(word)

        return tokens

    def tokenize(self, ids, index, path):
        # Tokenize file content
        with open(path, 'r') as f:
            token = 0
            for line in f:
                words = line.split() + ['<eos>']
                for word in words:
                    ids[index + token] = self.dictionary.word2idx[word]
                    token += 1

        return token
