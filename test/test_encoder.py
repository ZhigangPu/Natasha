from encoders.image_encoder import ImageEncoder
import torch
from utils.config import Config


def test_encoder(config):
    encoder = ImageEncoder(config)
    X = torch.randn(4, 3, 256, 256)
    dec_init_state, visual_context= encoder.forward(X)
    assert visual_context.size() == torch.Size([4, 1056, 1024])
    assert dec_init_state[0].size() == torch.Size([4, 512])


if __name__ == '__main__':
    test_encoder(Config('../config/encoder.config.test.json'))
