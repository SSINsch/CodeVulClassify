import torch
import psutil
import json
import collections
from typing import List, Tuple, Dict
from transformers import BertTokenizer

from parser_tokenizers import CodeTokenizer, CodeAbsTokenizer
import logging

logger = logging.getLogger(__name__)


def load_tokenizer(tokenizer_type):
    if tokenizer_type == 'bert-base-uncased':
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    elif tokenizer_type == 'code-parser':
        tokenizer = CodeTokenizer(lang='c')
    elif tokenizer_type == 'code-abs-parser':
        tokenizer = CodeAbsTokenizer(lang='c')
    else:
        raise ValueError(f'Unknown tokenizer type:{tokenizer_type}. Type must be (bert-base-uncased / code-parser).')

    return tokenizer


def load_config(path_config: str):
    logger.info(f'config file : {path_config}')
    try:
        with open(path_config, 'r') as json_file:
            json_data = json.load(json_file)
    except Exception as e:
        logger.error(f'Error while loading config file!')
        logger.error(e)
        raise ValueError(f'Error while loading config file!')

    path_train = json_data['DATA']['TRAIN']
    path_valid = json_data['DATA']['VALID']
    path_test = json_data['DATA']['TEST']
    hidden_dim = json_data['hidden_dimension_size']
    embed_size = json_data['embedding_size']
    num_lstm_layer = json_data['number_of_lstm_layer']
    batch_size = json_data['batch_size']
    n_classes = json_data['number_of_label_class']
    lr = json_data['learning_rate']
    n_epoch = json_data['number_of_epoch']
    tokenizer = json_data['tokenizer']
    seed = json_data['random_seed']
    model = json_data['model']
    block_size = json_data['block_size']
    output_dir = json_data['output_dir']

    return [path_train, path_valid, path_test, hidden_dim, embed_size, num_lstm_layer, batch_size, n_classes, lr,
            n_epoch, tokenizer, seed, model, block_size, output_dir]


def read_jsonl_data(data_path: str):
    """
    jsonl 파일을 읽어서 return
    """
    json_data = []
    with open(data_path, 'r') as f:
        for line in f:
            json_line = json.loads(line.strip())
            json_data.append(json_line)

    return json_data


def build_tok_vocab(tokenize_target: List, tokenizer,
                    min_freq: int = 3, max_vocab=29998) -> Tuple[List[str], Dict]:
    """
    데이터 입력 받아서 vocab set return
    :param tokenize_target:
    :param tokenizer:
    :param min_freq: vocab set 최소 회수
    :param max_vocab: vocab set 최대 크기
    :return: (단어, idx)으로 이루어진 Tuple
    """
    vocab = []
    logger.debug('start parsing vocabulary set!')
    for i, target in enumerate(tokenize_target):
        try:
            temp = tokenizer.tokenize(target)
            vocab.extend(temp)

        except Exception as e_msg:
            error_target = f'idx: {i} \t target:{target}'
            logger.warning(error_target, e_msg)

    logger.debug('vocabulary set parsing done')
    logger.debug('start configuring vocabulary set!')
    vocab = collections.Counter(vocab)
    temp = {}
    # min_freq보다 적은 단어 거르기
    for key in vocab.keys():
        if vocab[key] >= min_freq:
            temp[key] = vocab[key]
    vocab = temp

    # 가장 많이 등장하는 순으로 정렬한 후, 적게 나온것 위주로 vocab set에서 빼기
    vocab = sorted(vocab, key=lambda x: -vocab[x])
    if len(vocab) > max_vocab:
        vocab = vocab[:max_vocab]

    tok2idx = {'<pad>': 0, '<unk>': 1}
    for tok in vocab:
        tok2idx[tok] = len(tok2idx)
    vocab.extend(['<pad>', '<unk>'])

    logger.debug('vocabulary set configuring done')

    return vocab, tok2idx


def to_np(x):
    return x.detach().cpu().numpy()


def prepare_sequence(seq, word_to_idx):
    indices = []

    for word in seq:
        if word not in word_to_idx.keys():
            indices.append(word_to_idx['<unk>'])
        else:
            indices.append(word_to_idx[word])

    # indices = np.array(indices)

    return indices


def debug_memory():
    print(f'\t memory allocated: {torch.cuda.memory_allocated()}, {torch.cuda.max_memory_allocated()}')
    print(f'\t memory reserved: {torch.cuda.memory_reserved()}, {torch.cuda.max_memory_reserved()}')
    print(f'\t RAM: {psutil.virtual_memory().percent}')


"""
def debug_result(pred, argmax_pred, argmax_answer, sent, debug_output, heatmap_name):
    pred = np.array(pred).T
    complete_sent = []
    for s in sent:
        complete_sent.append(''.join(s))
    col = ['answer', 'pred', 'prob_none', 'prob_offe', 'prob_hate', 'sentence']
    d = {'answer': replace_num(argmax_answer), 'pred': replace_num(argmax_pred), 'prob_none': pred[0],
         'prob_offe': pred[1], 'prob_hate': pred[2], 'sentence': complete_sent}
    df = pd.DataFrame(data=d, columns=col)
    df_wrong = df[df['answer'] != df['pred']]
    df.to_csv(debug_output, sep='\t')
    df_wrong.to_csv(debug_output.replace('.tsv', '_wrong.tsv'), sep='\t')

    plt.figure()
    cf = confusion_matrix(argmax_answer, argmax_pred)
    sns.heatmap(cf, annot=True, cmap='Blues')
    plt.savefig(heatmap_name)


def plot_att_val(sents, att_vals, predictions, labels):
    for i in range(len(sents)):
        x = sents[i]
        y = to_np(att_vals[i])
        if len(x) < 71:
            temp = [' ']*(71-(len(x)))
            x.extend(temp)
            temp = [0] * (71 - (len(x)))
            temp = np.array(temp)
            y = np.concatenate([y, temp])

        df = pd.DataFrame({"att_val": y},
                          index=x)
        plt.figure(figsize=(15, 15))
        sns.heatmap(df, fmt="g", cmap='viridis')
        name = f'./log/{i:04d}_{labels[i]}_{predictions[i]}.png'
        plt.savefig(name)
"""
