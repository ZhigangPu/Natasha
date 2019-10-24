from decoders.decoder import RNNDecoder
from embedder.embedding import Embeddings
from embedder.vocab import Vocab
from utils.config import Config
from encoders.image_encoder import ImageEncoder
import torch
from embedder.vocab import read_corpus, pad_sents


def test_decoder(config, dec_init_state, visual_context, target_padded):
    vocab = Vocab.load(config.vocab_path)
    embedding = Embeddings(config, vocab)
    decoder = RNNDecoder(config,
                         embedding=embedding,
                         dec_init_state=dec_init_state,
                         visual_context=visual_context,
                         target_padded=target_padded)

    combined_output = decoder.forward()
    print(combined_output.size())


if __name__ == '__main__':
    config_decoder = Config("../config/decoder.config.test.json")
    config_encoder = Config("../config/encoder.config.test.json")

    X = torch.randn(5, 3, 256, 256)
    encoder = ImageEncoder(config_encoder)
    dec_init_state, visual_context = encoder.forward(X)

    target_padded = pad_sents(read_corpus("../data/train.formulas.norm.small.txt", source='tgt'), pad_token=0)

    test_decoder(config_decoder, dec_init_state, visual_context, target_padded)
