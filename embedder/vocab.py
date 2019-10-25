import click
import json
import torch
from collections import Counter
from itertools import chain


def pad_sents(sents, pad_token):
    """ Pad list of sentences according to the longest sentence in the batch.

    Args:
        sents (list[list[str]]): list of sentences, where each sentence
                                    is represented as a list of words
        pad_token (str): padding token

    Returns:
        sents_padded (list[list[str]]): list of sentences where sentences shorter
        than the max length sentence are padded out with the pad_token, such that
        each sentences in the batch now has equal length.
    """
    sents_padded = []

    max_length = sorted([len(sentence) for sentence in sents])[-1]
    for sentence in sents:
        sentence += [pad_token] * (max_length - len(sentence))
        sents_padded.append(sentence)

    return sents_padded


def read_corpus(file_path, source):
    """ Read file, where each sentence is dilineated by a `\n`.

    Args:
        file_path (str): path to file containing corpus
        source (str): "tgt" or "src" indicating whether text
        is of the source language or target language

    Returns:
        data(List[List[str]]: list of list of words
    """
    data = []
    for line in open(file_path):
        sent = line.strip().split(' ')
        # only append <s> and </s> to the target sentence
        if source == 'tgt':
            sent = ['<s>'] + sent + ['</s>']
        data.append(sent)

    return data


class VocabEntry:
    """Vocabulary entry"""
    def __init__(self, word2id=None):
        """Init VocabEntry Instance

        Args:
            word2id(dict: str->int): dictionary mapping words to indices
        """
        if word2id:
            self.word2id = word2id
        else:
            self.word2id = dict()
            self.word2id['<pad>'] = 0
            self.word2id['<s>'] = 1
            self.word2id['</s>'] = 2
            self.word2id['<unk>'] = 3
        self.unk_id = self.word2id['<unk>']
        self.id2word = {v: k for k, v in self.word2id.items()}

    def __getitem__(self, word):
        """Retrieve word's index. Return the index for the unk
        if the word is out of vocabulary.

        Args:
            word(str): word to look up

        Returns:
            index(int): index of word
        """
        return self.word2id.get(word, self.unk_id)

    def __contains__(self, word):
        """Check if word is captured by VocabEntry

        Args:
            word(str): word to look up

        Returns:
            contains(bool): whether word is contained
        """
        return word in self.word2id

    def __len__(self):
        """Computer number of words in VocabEntry

        Returns:
            len(int): number of words here
        """
        return len(self.word2id)

    def __repr__(self):
        """Representation of VocabEntry to be used when printing the objects"""
        return 'Vocabulary[size=%d]' % len(self)

    def id2word(self, word_idx):
        """Return mapping of index to word.

        Args:
            word_idx(int): word index

        Returns：
            word(str): word corresponding to index
        """
        return self.id2word[word_idx]

    def add(self, word):
        """Add word to VocabEntry, if it is previously unseen

        Args:
            word(str): word to be added

        Returns:
            word_index(int): index that the word has been assigned
        """
        if word not in self.word2id:
            word_index = self.word2id[word] = len(self)
            self.id2word[word_index] = word
            return word_index
        else:
            return self[word]

    def words2indices(self, sents):
        """Convert list of words or list of sentences of words into list or list of indices

        Args:
            sents:(list[str] or list[list[str]]): sentences in words

        Returns:
            word_ids (list[int] or list[list[int]]: sentences in indices
        """
        if type(sents[0]) == list:
            return [[self[w] for w in s] for s in sents]
        else:
            return [self[w] for w in sents]

    def indices2words(self, word_ids):
        """Convert list of indices into word

        Args:
            word_ids(list[int]): list of word ids

        Returns:
            sents(list[str]): list of words
        """
        return [self.id2word[word_id] for word_id in word_ids]

    def to_input_tensor(self, sents, device):
        """Convert list of sentences (words) into tensor with necessary padding for
        shorter sentences

        Args:
            sents(List[List[str]]): list of sequences (words)
            device: device on which to load the tensor, i.e. CPU or GPU

        Returns:
            sents_tensor: tensor of (max_sentence_length, batch_size)
        """
        word_ids = self.words2indices(sents)
        sents_pad = pad_sents(word_ids, self['<pad>'])
        sents_tensor = torch.tensor(sents_pad, dtype=torch.long, device=device)   # (seq_len, batch_size)

        return torch.t(sents_tensor)  # (batch_size, seq_len)

    @staticmethod
    def from_corpus(corpus, size, freq_cutoff=2):
        """Given a corpus to construct a Vocab Entry

        Args:
            corpus(List[List[str]]): corpus of text produced by read_copus function
            size(int): number of words in vocabulary
            freq_cutoff: if word occurs n < freq_cutoff times, drop the word

        Returns:
            vocab_entry(VocabEntry): VocabEntry instance produced from previous corpus
        """
        vocab_entry = VocabEntry()
        word_freq = Counter(chain(*corpus))
        valid_words = [w for w, v in word_freq.items() if v >= freq_cutoff]
        print('number of word types: {}, number of word types w/ frequency >= {}: {}'
              .format(len(word_freq), freq_cutoff, len(valid_words)))
        top_k_words = sorted(valid_words, key=lambda w: word_freq[w], reverse=True)[:size]
        for word in top_k_words:
            vocab_entry.add(word)
        return vocab_entry


class Vocab:
    """Vocab encapsulating src and target languages"""

    def __init__(self, src_vocab, tgt_vocab):
        """Init vocab

        Args:
            src_vocab (VocabEntry): vocab entry for source language, maybe none if no source sentences
            tgt_vocab (VocabEntry): vocab entry for target language
        """
        self.src = src_vocab
        self.tgt = tgt_vocab

    @staticmethod
    def build(src_sents, tgt_sents, vocab_size, freq_cutoff):
        """ Build Vocabulary.

        Args:
            src_sents (list[str]): Source sentences provided by read_corpus() function
            tgt_sents (list[str]): Target sentences provided by read_corpus() function
            vocab_size (int): Size of vocabulary for both source and target languages
            freq_cutoff (int): if word occurs n < freq_cutoff times, drop the word.
        """
        if src_sents is not None and tgt_sents is not None:
            assert len(src_sents) == len(tgt_sents)

        print('initialize source vocabulary ..')
        src = VocabEntry.from_corpus(src_sents, vocab_size, freq_cutoff) if src_sents is not None else None

        print('initialize target vocabulary ..')
        tgt = VocabEntry.from_corpus(tgt_sents, vocab_size, freq_cutoff) if tgt_sents is not None else None

        return Vocab(src, tgt)

    def save(self, file_path):
        """ Save Vocab to file as JSON dump.

        Args:
            file_path (str): file path to vocab file
        """
        if self.src is not None:
            json.dump(dict(src_word2id=self.src.word2id, tgt_word2id=self.tgt.word2id), open(file_path, 'w'), indent=2)
        else:
            json.dump(dict(tgt_word2id=self.tgt.word2id), open(file_path, 'w'), indent=2)

    @staticmethod
    def load(file_path):
        """ Load vocabulary from JSON dump.

        Args:
            file_path (str): file path to vocab file
        Returns:
            Vocab object loaded from JSON dump
        """
        entry = json.load(open(file_path, 'r'))
        if 'src_word2id' in entry:
            src_word2id = entry['src_word2id']
            tgt_word2id = entry['tgt_word2id']
            return Vocab(VocabEntry(src_word2id), VocabEntry(tgt_word2id))
        else:
            tgt_word2id = entry['tgt_word2id']
            return Vocab(None, VocabEntry(tgt_word2id))

    def __repr__(self):
        """ Representation of Vocab to be used
        when printing the object.
        """
        return 'Vocab(source %d words, target %d words)' % (len(self.src), len(self.tgt))


@click.command()
@click.option('--train_src', default="data/train_source_sents.txt")
@click.option('--train_tgt', default="data/train_target_sents.txt")
@click.option('--vocab_size', default=100000)
@click.option('--freq_cutoff', default=2)
@click.option('--vocab_path', default="data/vocab_train.txt")
def main(train_src, train_tgt, vocab_size, freq_cutoff, vocab_path):

    print('read in source sentences: %s' % train_src)
    print('read in target sentences: %s' % train_tgt)

    if train_src == 'none':
        src_sents = None
    else:
        src_sents = read_corpus(train_src, source='src')
    tgt_sents = read_corpus(train_tgt, source='tgt')

    vocab = Vocab.build(src_sents, tgt_sents, int(vocab_size), int(freq_cutoff))
    print('generated vocabulary, source %d words, target %d words' % (len(vocab.src) if vocab.src is not None else 0,
                                                                      len(vocab.tgt)))

    vocab.save(vocab_path)
    print('vocabulary saved to %s' % vocab_path)


if __name__ == '__main__':
    main()
