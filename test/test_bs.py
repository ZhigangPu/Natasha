from model.model import ImageToLatexModel
from decoders.decoder import RNNDecoder
from encoders.image_encoder import ImageEncoder
from utils.config import Config
from embedder.vocab import Vocab
from embedder.embedding import Embeddings
from inputer.data_generator import DataGenerator


def main():
    """config"""
    config_decoder = Config("../config/decoder.config.test.json")
    config_encoder = Config("../config/encoder.config.test.json")

    """dataset"""
    path_formulas = "../data/train.formulas.norm.txt"
    dir_images = "../data/images_train/"
    path_matching = "../data/train.matching.txt"
    train_set = DataGenerator(
        path_formulas=path_formulas,
        dir_images=dir_images,
        path_matching=path_matching,
        max_iter=1000,
        bucket=True,
        bucket_size=4
    )

    """encoder"""
    encoder = ImageEncoder(config_encoder)

    """decoder"""
    vocab = Vocab.load(config_decoder.vocab_path)
    embedding = Embeddings(config_decoder, vocab)
    decoder = RNNDecoder(config_decoder,
                         embedding=embedding)

    """train"""
    model = ImageToLatexModel(encoder, decoder)
    for sample in train_set:
        image = sample[0]
        hypothesis = model.beam_search([image])
        print(len(hypothesis))
        break




if __name__ == '__main__':
    main()
