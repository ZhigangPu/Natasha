import torch.nn.functional as F
import torch.nn as nn
import torch


class ImageEncoder:
    """Image encoder with CNN -> RNN architecture

    Attributes:
        num_image_channel: image color channel
        vocab_size: vocabulary size
        hidden_size: rnn encoder hidden size
        num_rnn_layers: rnn encoder hidden layer
        num_directions: if bidirectional: 2, else 1
        dropout: rnn dropout rate
        h_projection: last rnn hidden state cast transformation, \
                    for sake of init hidden state of decoder
        c_projection: last rnn cell state cast transformation, \
                    for sake of init cell state of decoder
    """
    def __init__(self, config):
        # cnn
        self.num_image_channel = config.num_image_channel
        # rnn
        self.hidden_size = config.hidden_size
        self.num_rnn_layers = config.num_rnn_layers
        self.num_directions = 2 if config.bidirectional else 1
        self.dropout = config.dropout if config.dropout is not False else None
        self.h_projection = nn.Linear(self.num_directions * self.hidden_size, self.hidden_size, bias=False)
        self.c_projection = nn.Linear(self.num_directions * self.hidden_size, self.hidden_size, bias=False)

        # cnn architecture
        self.layer1 = nn.Conv2d(self.num_image_channel, 64, kernel_size=(3, 3),
                                padding=(1, 1), stride=(1, 1))
        self.layer2 = nn.Conv2d(64, 128, kernel_size=(3, 3), padding=(1, 1),
                                stride=(1, 1))
        self.layer3 = nn.Conv2d(128, 256, kernel_size=(3, 3), padding=(1, 1),
                                stride=(1, 1))
        self.layer4 = nn.Conv2d(256, 256, kernel_size=(3, 3), padding=(1, 1),
                                stride=(1, 1))
        self.layer5 = nn.Conv2d(256, 512, kernel_size=(3, 3), padding=(1, 1),
                                stride=(1, 1))
        self.layer6 = nn.Conv2d(512, 512, kernel_size=(3, 3), padding=(1, 1),
                                stride=(1, 1))

        self.batch_norm1 = nn.BatchNorm2d(256)
        self.batch_norm2 = nn.BatchNorm2d(512)
        self.batch_norm3 = nn.BatchNorm2d(512)

        self.rnn = nn.LSTM(input_size=512,
                           hidden_size=self.hidden_size,
                           num_layers=self.num_rnn_layers,
                           bias=False,
                           dropout=self.dropout if self.dropout is not None else 0,
                           bidirectional=True if self.num_directions == 2 else False)

        self.embedding = nn.Embedding(500, 512)  # here vocab_size equals to row numbers in the last cnn layer

    def forward(self, src):
        """Forward operation of encoder

        Args:
            src: tensor of batch images with size (batch_size, channels, height, weight)

        Returns:
            visual_context: tensor, containing last LSTM hidden layer across every rows of
                            cnn feature maps. See more with 'Image-to-Markup Generation with
                            Coarse-to-Fine Attention'

            dec_init_state: tuple. (last_enc_hidden_t, last_enc_cell_t)
        """

        # layer 1
        # (batch_size, 64, imgH, imgW)
        # TODO： harvard nlp's implementation is 'self.layer1(src[:, :, :, :] - 0.5)', motivation?
        src = F.relu(self.layer1(src), True)

        # (batch_size, 64, imgH/2, imgW/2)
        src = F.max_pool2d(src, kernel_size=(2, 2), stride=(2, 2))

        # layer 2
        # (batch_size, 128, imgH/2, imgW/2)
        src = F.relu(self.layer2(src), True)

        # (batch_size, 128, imgH/2/2, imgW/2/2)
        src = F.max_pool2d(src, kernel_size=(2, 2), stride=(2, 2))

        # layer 3
        # (batch_size, 256, imgH/2/2, imgW/2/2)
        src = F.relu(self.batch_norm1(self.layer3(src)), True)

        # layer4
        # (batch_size, 256, imgH/2/2, imgW/2/2)
        src = F.relu(self.layer4(src), True)

        # (batch_size, 256, imgH/2/2/2, imgW/2/2)
        src = F.max_pool2d(src, kernel_size=(1, 2), stride=(1, 2))

        # layer5
        # (batch_size, 512, imgH/2/2/2, imgW/2/2)
        src = F.relu(self.batch_norm2(self.layer5(src)), True)

        # (batch_size, 512, imgH/2/2/2, imgW/2/2/2)
        src = F.max_pool2d(src, kernel_size=(2, 1), stride=(2, 1))

        # layer6
        # (batch_size, 512, imgH/2/2/2, imgW/2/2/2)
        src = F.relu(self.batch_norm3(self.layer6(src)), True)

        # rnn encoder
        visual_context = []
        enc_last_states = None
        for row in range(src.size(2)):
            input_from_cnn_features = src[:, :, row, :].transpose(0, 2).transpose(1, 2)  # (imgW/2/2/2, batch_size, 512)
            input_to_add_h0 = torch.Tensor(src.size(0)).type_as(input_from_cnn_features)\
                .long().fill_(row)  # (batch_size)
            input_to_add_h0 = self.embedding(input_to_add_h0)
            input_rnn = torch.cat(
                (input_to_add_h0.view(1, input_to_add_h0.size(0), input_to_add_h0.size(1)), input_from_cnn_features), 0
                )
            enc_hiddens, enc_last_states = self.rnn(input_rnn)  # enc_hiddens: (imgW/2/2/2 + 1, batch_size, hidden_size)
            visual_context.append(enc_hiddens)
        # visual_context: (batch_size, imgH/2/2/2 * (imgW/2/2/2 + 1), hidden_size)
        visual_context = torch.cat(visual_context, 0).transpose(0, 1)

        # last_hidden_t: (bidirection, batch_size, hidden_size)
        last_hidden, last_cell = enc_last_states
        last_hidden = torch.cat((last_hidden[0], last_hidden[1]), 1)
        dec_init_hidden = self.h_projection(last_hidden)   # dec_init_hidden: (batch_size, hidden_size)
        last_cell = torch.cat((last_cell[0], last_cell[1]), 1)
        dec_init_cell = self.c_projection(last_cell)
        dec_init_state = (dec_init_hidden, dec_init_cell)

        return dec_init_state, visual_context
