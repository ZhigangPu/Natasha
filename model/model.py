from utils.image_operation import pad_batch_images
from embedder.vocab import pad_sents
import torch.nn as nn
import torch


class ImageToLatexModel(nn.Module):
    """Model translating images to latex code"""

    def __init__(self, encoder, decoder):
        super(ImageToLatexModel, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, batch):
        """ Take a mini-batch of source image and target sentences, computed the log-likelihood
        of target sentences under the image to latex models

        Args:
            batch(tuple): (list[image], list[formula str])

        Returns:
            scores(tensor): log-likelihood of generating true target words
        """
        images, formulas = batch
        batch_size = len(images)
        images_padded = pad_batch_images(images)
        images_t = torch.tensor(images_padded).transpose(1, 3).transpose(2, 3).float()
        formulas_padded = pad_sents(formulas, '<pad>')

        dec_init_state, visual_context = self.encoder(images_t)
        scores = self.decoder(dec_init_state, visual_context, formulas_padded)

        sample_loss = -scores
        batch_loss = sample_loss.sum()
        loss = batch_loss / batch_size

        return loss

    def beam_search(self, image):
        """Given a single source sentence, perform beam search, yielding translation in the target

        Args:
            image: source image (1, channel, h, w)

        Returns:
            hypothesis(List(Hypothesis)): a list of hypothesis. see more at decoder class definition
        """
        image_padded = pad_batch_images(image)
        image_t = torch.tensor(image_padded).transpose(1, 3).transpose(2, 3).float()
        dec_init_state, visual_context = self.encoder(image_t)

        hypothesis = self.decoder.beam_search(dec_init_state, visual_context, beam_size=5, max_decoding_time_step=150)

        return hypothesis
