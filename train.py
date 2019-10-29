import sys
import math
import time
import click
import torch
import numpy as np
from utils.config import Config
from utils.linux_wrapper import delete_file
from inputer.data_generator import DataGenerator
from embedder.vocab import Vocab
from embedder.embedding import Embeddings
from encoders.image_encoder import ImageEncoder
from decoders.decoder import RNNDecoder
from model.model import ImageToLatexModel


def evaluate_ppl(model, val_set, batch_size=10):
    """ Evaluate perplexity on validation sentences
    
    Args:
        model(nn.Module): ImageToLatexModel
        val_data(list[(image, formula)]): list of tuples containing validation image and formulas 
        batch_size(int): batch size
    
    Returns:
        ppl(float): perplexity on validation set
    """
    was_training = model.training
    model.eval()

    cum_loss = 0
    cum_tgt_words = 0

    with torch.no_grad():
        for batch in val_set.minibatch(batch_size):
            loss = model(batch)
            cum_loss += loss.item() * batch_size
            target_words_num = sum(len(tgt[1:]) for tgt in batch[1])
            cum_tgt_words += target_words_num
            break

        ppl = np.exp(cum_loss / cum_tgt_words)

    if was_training:
        model.train()

    return ppl, loss


def train(config_train):
    # load config
    config_encoder = Config(config_train.config_encoder)
    config_decoder = Config(config_train.config_decoder)

    # load training set and validation set (or development set)
    train_set = DataGenerator(
        path_formulas=config_train.train_path_formulas,
        dir_images=config_train.train_dir_images,
        path_matching=config_train.train_path_matching,
    )
    val_set = DataGenerator(
        path_formulas=config_train.val_path_formulas,
        dir_images=config_train.val_dir_images,
        path_matching=config_train.val_path_matching
    )

    batch_size = config_train.batch_size
    clip_grad = config_train.clip_grad
    valid_steps = config_train.valid_steps
    log_steps = config_train.log_steps
    model_save_path = config_train.save_path

    # delele old archives
    delete_file(config_train.save_path+'train_loss.txt')
    delete_file(config_train.save_path+'val_loss.txt')

    encoder = ImageEncoder(config_encoder)
    vocab = Vocab.load(config_decoder.vocab_path)
    embedding = Embeddings(config_decoder, vocab)
    decoder = RNNDecoder(config_decoder, embedding)
    model = ImageToLatexModel(encoder, decoder)

    model.train()

    # initializing parameter with uniform distribution
    uniform_unit = config_train.uniform_unit
    if np.abs(uniform_unit) > 0:
        print('uniformly initializing parameters [-%f, +%f]' %  (uniform_unit, uniform_unit), file=sys.stdout)
        for p in model.parameters():
            p.data.uniform_(-uniform_unit, uniform_unit)

    device = torch.device('cuda:0' if config_train.cuda else 'cpu')
    print('use device: %s' % device, file=sys.stdout)

    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config_train.learning_rate)

    # number of trial to avoid overfitting, if reaching max, early stop!
    num_trial = 0
    train_iter = patience = cum_loss = report_loss = cum_tgt_words = report_tgt_words = 0
    cum_examples = report_examples = epoch = valid_num = 0
    hist_valid_scores = []
    train_time = begin_time = time.time()
    print('begin training ...')

    while True:
        epoch += 1

        for batch in train_set.minibatch(batch_size):
            train_iter += 1
            optimizer.zero_grad()

            loss = model(batch)

            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)

            optimizer.step()

            batch_loss = loss.item() * batch_size
            report_loss += batch_loss
            cum_loss += batch_loss

            target_words_num = sum(len(tgt[1:]) for tgt in batch[1])
            report_tgt_words += target_words_num
            cum_tgt_words += target_words_num
            report_examples += batch_size
            cum_examples += batch_size

            if train_iter % log_steps == 0:
                print('epoch %d, iter %d, avg. loss %.2f, avg. ppl %.2f ' \
                      'cum. examples %d, speed %.2f words/sec, time elapsed %.2f sec' % (epoch, train_iter,
                                                                                         report_loss / report_examples,
                                                                                         math.exp(report_loss / report_tgt_words),
                                                                                         cum_examples,
                                                                                         report_tgt_words / (time.time() - train_time),
                                                                                         time.time() - begin_time), file=sys.stdout)
                train_time =time.time()
                report_loss = report_examples = report_tgt_words = 0.

            if train_iter % valid_steps == 0:
                print('epoch %d, iter %d, cum. loss %.2f, cum. ppl %.2f cum. examples %d' % (epoch, train_iter,
                                                                                             cum_loss / cum_examples,
                                                                                             np.exp(cum_loss / cum_tgt_words),
                                                                                             cum_examples), file=sys.stdout)
                with open(config_train.save_path + 'train_loss.txt', 'a') as f:
                    f.write(str(cum_loss / cum_examples) + '\n')
                cum_loss = cum_examples = 0.
                valid_num += 1

                print('begin validation ...', file=sys.stdout)

                # compute val. ppl and belu
                val_ppl, val_loss = evaluate_ppl(model, val_set, batch_size=100)
                valid_metric = -val_ppl
                with open(config_train.save_path + 'val_loss.txt', 'a') as f:
                    f.write(str(val_loss.data.item()) + '\n')

                print('validation: iter %d, val. ppl %f' % (train_iter, val_ppl), file=sys.stdout)

                is_better = len(hist_valid_scores) == 0 or valid_metric > max(hist_valid_scores)
                hist_valid_scores.append(valid_metric)

                if is_better:
                    patience = 0
                    print('save currently the best model to [%s]' % model_save_path, file=sys.stdout)
                    model.save(model_save_path, 'best.model')

                    torch.save(optimizer.state_dict(), model_save_path + '.optim')
                elif patience < config_train.patience:
                    patience += 1
                    print('hit patience %d' % patience, file=sys.stdout)

                    if patience == config_train.patience:
                        num_trial += 1
                        print('hit #%d trial' % num_trial, file=sys.stdout)
                        if num_trial == config_train.max_trial:
                            print('eraly stop!', file=sys.stdout)
                            sys.exit(0)

                        # decay lr, and restore from previously best checkpoint
                        # TODO: add more refined learning rate scheduler
                        lr = optimizer.param_groups[0]['lr'] * float(config_train.lr_decay)
                        print('load previously best model and decay learning rate to %f' % lr, file=sys.stdout)

                        # load model
                        params = torch.load(model_save_path + 'best.model', map_location=lambda storage, loc: storage)
                        model.load_state_dict(params['state_dict'])
                        model = model.to(device)

                        print('restore parameters of the optimizers', file=sys.stdout)
                        optimizer.load_state_dict(torch.load(model_save_path + '.optim'))

                        # set new lr
                        for param_group in optimizer.param_groups:
                            param_group['lr'] = lr

                        # reset patience
                        patience = 0

                if epoch == config_train.max_epoch:
                    print('reach maximum number of epoches!', file=sys.stdout)
                    sys.exit(0)


@click.command()
@click.option("--train_config", default='./config/train.json')
def main(train_config):
    config_train = Config(train_config)
    train(config_train)


if __name__ == '__main__':
    main()