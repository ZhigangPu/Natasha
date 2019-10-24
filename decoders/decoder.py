import torch.nn as nn
import torch
import torch.nn.functional as F


class RNNDecoder:
    """LSTM + Attention (+ coverage + copy)

    Attributes:
        embedding: target sentences word embedding object
        dropout: dropout layer
        hidden_size: hidden_size of a LSTM cell
        dec_init_state: last encoder state, tuple, each size (direction, batch_size, enc_hidden_size)
        visual_context: context matrix for attention mechanism (batch_size, seq_len, 2 * enc_hidden_size)
        target_padded: padded target sentences, (tgt_len, b)
        att_projection: W_{attProj}
        combined_output_projection: W_u
        target_vocab_projection: W_{vocab}
        decoder: LSTM cell performing an update step
    """

    def __init__(self, config, embedding, dec_init_state, visual_context, target_padded):
        self.embedding = embedding
        self.dropout = nn.Dropout(config.dropout)
        self.hidden_size = config.hidden_size
        self.dec_init_state = dec_init_state
        self.visual_context = visual_context
        self.target_padded = target_padded
        self.att_projection = nn.Linear(2*self.hidden_size, self.hidden_size)
        self.combined_output_projection = nn.Linear(3 * self.hidden_size, self.hidden_size, bias=False)
        self.target_vocab_projection = nn.Linear(self.hidden_size, self.embedding.target.embedding_dim)
        self.decoder = nn.LSTMCell(
            self.embedding.embed_size + self.hidden_size,
            self.hidden_size,
            bias=False
        )

    def forward(self):
        """Compute all combined output of each timestep"""
        target_padded_t = self.embedding.vocab.tgt.to_input_tensor(self.target_padded, torch.device('cpu'))
        target_padded_t = target_padded_t[:-1]  # Chop of the <END> token for max length sentences, TODO why chopping?

        # Initialize combine output vector as zero
        batch_size = self.visual_context.size(0)
        o_prev = torch.zeros(batch_size, self.hidden_size)

        # Initialize decoder state
        dec_state = self.dec_init_state

        # Initialize a list we will use to collect the combined output o_t on each step
        combined_output = []

        enc_hidden_proj = self.att_projection(self.visual_context)  # (batch_size, seq_len, hidden_size)
        Y = self.embedding.target(target_padded_t)

        for Y_t in torch.split(Y, 1, dim=0):  # (1, b ,e)
            Y_t = torch.squeeze(Y_t)   # (b, e)
            Ybar_t = torch.cat((o_prev, Y_t), dim=1)
            dec_state, o_t, e_t = self.step(Ybar_t, dec_state, enc_hidden_proj)

            combined_output.append(o_t)
            o_prev = o_t

        combined_output = torch.stack(combined_output)

        return combined_output

    def step(self, Ybar_t, dec_state, enc_hidden_proj):
        """One step of decoder"""

        dec_state = self.decoder(Ybar_t, dec_state)
        dec_hidden, dec_cell = dec_state  # dec_hidden (batch_size, h)
        e_t = torch.squeeze(
            torch.bmm(enc_hidden_proj, torch.unsqueeze(dec_hidden, 2)), 2
        )

        alpha_t = F.softmax(e_t, dim=1)  # alpha_t (batch_size, seq_len)
        # a_t (batch_size, 1, 2 * hidden_size)
        a_t = torch.squeeze(torch.bmm(torch.unsqueeze(alpha_t, 1), self.visual_context))

        u_t = torch.cat((a_t, dec_hidden), 1)  # u_t (batch_size, 3 * hidden_size)
        v_t = self.combined_output_projection(u_t)  # v_t (batch_size, hidden_size)
        o_t = self.dropout(torch.tanh(v_t))

        combined_output = o_t
        return dec_state, combined_output, e_t
