import torch.nn as nn
import torch
import torch.nn.functional as F

from collections import namedtuple
Hypothesis = namedtuple('Hypothesis', ['value', 'score'])

class RNNDecoder(nn.Module):
    """LSTM + Attention (+ coverage + copy)

    Attributes:
        embedding: target sentences word embedding object
        dropout: dropout layer
        hidden_size: hidden_size of a LSTM cell
        att_projection: W_{attProj}
        combined_output_projection: W_u
        target_vocab_projection: W_{vocab}
        decoder: LSTM cell performing an update step
    """

    def __init__(self, config, embedding):
        super(RNNDecoder, self).__init__()
        self.config = config
        self.embedding = embedding
        self.dropout = nn.Dropout(config.dropout)
        self.hidden_size = config.hidden_size
        self.att_projection = nn.Linear(2*self.hidden_size, self.hidden_size)
        self.combined_output_projection = nn.Linear(3 * self.hidden_size, self.hidden_size, bias=False)
        self.target_vocab_projection = nn.Linear(self.hidden_size, len(self.embedding.vocab.tgt))
        self.decoder = nn.LSTMCell(
            self.embedding.embed_size + self.hidden_size,
            self.hidden_size,
            bias=False
        )

    def forward(self, dec_init_state, visual_context, target_padded):
        """Compute all combined output of each timestep

        Args:
            dec_init_state: last encoder state, tuple, each size (direction, batch_size, enc_hidden_size)
            visual_context: context matrix for attention mechanism (batch_size, seq_len, 2 * enc_hidden_size)
            target_padded: padded target sentences, (tgt_len, b)

        Returns:
            P(tensor): Probability distribution over vocabulary (batch_size, vocab_size)
            scores(tensor): sum of gold word prediction accuracy
        """
        target_padded_t = self.embedding.vocab.tgt.to_input_tensor(target_padded, device=self.device)
        target_padded_t = target_padded_t[:-1]  # Chop of the <END> token for max length sentences, TODO why chopping?

        # Initialize combine output vector as zero
        batch_size = visual_context.size(0)
        o_prev = torch.zeros(batch_size, self.hidden_size, device=self.device)

        # Initialize decoder state
        dec_state = dec_init_state

        # Initialize a list we will use to collect the combined output o_t on each step
        combined_output = []

        enc_hidden_proj = self.att_projection(visual_context)  # (batch_size, seq_len, hidden_size)
        Y = self.embedding.target(target_padded_t)

        for Y_t in torch.split(Y, 1, dim=0):  # (1, b ,e)
            Y_t = torch.squeeze(Y_t)   # (b, e)
            Ybar_t = torch.cat((o_prev, Y_t), dim=1)
            dec_state, o_t, e_t = self.step(Ybar_t, dec_state, enc_hidden_proj, visual_context)

            combined_output.append(o_t)
            o_prev = o_t

        combined_output = torch.stack(combined_output)  # (seq_len, batch_size, hidden_size)
        P = F.log_softmax(self.target_vocab_projection(combined_output), dim=-1)  # (seq_len, batch_size, vocab_size)

        target_padded_indices = torch.tensor(self.embedding.vocab.tgt.words2indices(target_padded), device=self.device)\
            .transpose(0, 1)
        target_mask = (target_padded_indices != self.embedding.vocab.tgt['<pad>']).float()
        target_gold_word_log_pro = torch.gather(P, index=target_padded_indices[1:].unsqueeze(-1), dim=-1)\
                                       .squeeze(-1)*target_mask[1:]
        scores = target_gold_word_log_pro.sum(dim=0)
        return scores

    def step(self, Ybar_t, dec_state, enc_hidden_proj, visual_context):
        """One step of decoder

        Args:
            Ybar_t (Tensor): Concatenated Tensor of [Y_t o_prev], with shape (b, e + h). The input for the decoder,
                                where b = batch size, e = embedding size, h = hidden size.
            dec_state (tuple(Tensor, Tensor)): Tuple of tensors both with shape (b, h), where b = batch size, h = hidden size.
                First tensor is decoder's prev hidden state, second tensor is decoder's prev cell.
            enc_hiddens_proj (Tensor): Encoder hidden states Tensor, projected from (h * 2) to h. Tensor is with shape (b, src_len, h),
                                    where b = batch size, src_len = maximum source length, h = hidden size.
            visual_context(Tensor): Encoder hidden states Tensor, with shape (b, src_len, h * 2), where b = batch size,
                                    src_len = maximum source length, h = hidden size.
        Returns:
            dec_state (tuple (Tensor, Tensor)): Tuple of tensors both shape (b, h), where b = batch size, h = hidden size.
                First tensor is decoder's new hidden state, second tensor is decoder's new cell.
            combined_output (Tensor): Combined output Tensor at timestep t, shape (b, h), where b = batch size, h = hidden size.
            e_t (Tensor): Tensor of shape (b, src_len). It is attention scores distribution.
                                Note: You will not use this outside of this function.
                                      We are simply returning this value so that we can sanity check
                                      your implementation.
        """

        dec_state = self.decoder(Ybar_t, dec_state)
        dec_hidden, dec_cell = dec_state  # dec_hidden (batch_size, h)
        e_t = torch.squeeze(
            torch.bmm(enc_hidden_proj, torch.unsqueeze(dec_hidden, 2)), 2
        )

        alpha_t = F.softmax(e_t, dim=1)  # alpha_t (batch_size, seq_len)
        # a_t (batch_size, 1, 2 * hidden_size)
        a_t = torch.squeeze(torch.bmm(torch.unsqueeze(alpha_t, 1), visual_context), 1)

        u_t = torch.cat((a_t, dec_hidden), 1)  # u_t (batch_size, 3 * hidden_size)
        v_t = self.combined_output_projection(u_t)  # v_t (batch_size, hidden_size)
        o_t = self.dropout(torch.tanh(v_t))

        combined_output = o_t
        return dec_state, combined_output, e_t

    def beam_search(self, dec_init_state, visual_context, beam_size, max_decoding_time_step):
        """Given a single source sentence, perform beam search, yielding translations in the target languange.

        Args:
            dec_init_state: last encoder state, tuple, each size (direction, batch_size, enc_hidden_size)
            visual_context: context matrix for attention mechanism (batch_size, seq_len, 2 * enc_hidden_size)
            beam_size(int): beam size
            max_decoding_time_step(int): maximum number of time steps to unroll the decoding RNN

        Returns:
            hypothesis(list(hypothesis): a list of hypothesis, each hypothesis has two fields:
                value(list(str)): the decoded target sentences, represented as a list of words
                score(float): the log-likelihood of the target sentence
        """
        enc_hidden_proj = self.att_projection(visual_context)  # (batch_size, seq_len, hidden_size)

        h_prev = dec_init_state
        att_prev = torch.zeros(1, self.hidden_size, device=self.device)

        hyp_list = [['<s>']]
        hyp_scores = torch.zeros(len(hyp_list), dtype=torch.float, device=self.device)
        hyp_complete = []

        t = 0
        while len(hyp_complete) < beam_size and t < max_decoding_time_step:
            t += 1
            hyp_num = len(hyp_list)

            visual_context_ex = visual_context.expand(hyp_num, visual_context.size(1), visual_context.size(2))
            enc_hidden_proj_ex = enc_hidden_proj.expand(hyp_num, enc_hidden_proj.size(1), enc_hidden_proj.size(2))

            Y_t = torch.tensor([self.embedding.vocab.tgt[hyp[-1]] for hyp in hyp_list], dtype=torch.long, device=self.device)
            Ybar_t = self.embedding.target(Y_t)

            X = torch.cat([Ybar_t, att_prev], dim=-1)

            (h_t, c_t), att_t, _ = self.step(X, h_prev, enc_hidden_proj_ex, visual_context_ex)

            P = F.log_softmax(self.target_vocab_projection(att_t), dim=-1)

            hyp_tbd_num = beam_size - len(hyp_complete)
            continuaing_hyp_scores = (hyp_scores.unsqueeze(1).expand_as(P) + P).view(-1)
            topk_hyp_scores, topk_hyp_idx = torch.topk(continuaing_hyp_scores, k=hyp_tbd_num)

            # translate the topk result to acquire hypothesis id and word id
            hyp_ids = topk_hyp_idx / len(self.embedding.vocab.tgt)
            hyp_word_ids = topk_hyp_idx % len(self.embedding.vocab.tgt)

            hyp_list_new = []
            hyp_tbd_ids = []
            hyp_scores_new = []

            for hyp_id, hyp_word_id, hyp_score in zip(hyp_ids, hyp_word_ids, topk_hyp_scores):
                hyp_id = hyp_id.item()
                hyp_word_id = hyp_word_id.item()
                hyp_score = hyp_score.item()

                hyp_word = self.embedding.vocab.tgt.id2word[hyp_word_id]
                hyp_new_sent = hyp_list[hyp_id] + [hyp_word]

                if hyp_word == '</s>':
                    hyp_complete.append(Hypothesis(value=hyp_new_sent[1:-1], score=hyp_score))
                else:
                    hyp_list_new.append(hyp_new_sent)
                    hyp_tbd_ids.append(hyp_id)
                    hyp_scores_new.append(hyp_score)

            if len(hyp_complete) == beam_size:
                break

            hyp_tbd_ids_t = torch.tensor(hyp_tbd_ids, dtype=torch.long, device=self.device)
            h_prev = (h_t[hyp_tbd_ids], c_t[hyp_tbd_ids])
            att_prev = att_t[hyp_tbd_ids]

            hyp_list = hyp_list_new
            hyp_scores = torch.tensor(hyp_scores_new, dtype=torch.float, device=self.device)

        if len(hyp_complete) == 0:
            hyp_complete.append(Hypothesis(value=hyp_list[0][1:], score=hyp_scores[0].item()))

        hyp_complete.sort(key=lambda hyp: hyp.score, reverse=True)

        return hyp_complete

    @property
    def device(self) -> torch.device:
        """ Determine which device to place the Tensors upon, CPU or GPU.
        """
        return self.att_projection.weight.device
