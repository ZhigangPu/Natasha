import torch.nn as nn


class Embeddings(nn.Module):
    """Class storing embedding of source and target sequences

    Attributes:
        embed_size: embedding size
    """
    def __init__(self, config, vocab):
        super(Embeddings, self).__init__()
        self.embed_size = config.embed_size

        if vocab.src is not None:
            self.source = nn.Embedding(len(vocab.src), self.embed_size)
        if vocab.tgt is not None:
            self.target = nn.Embedding(len(vocab.tgt), self.embed_size)


