import random
import datetime
import torch
import torch.nn as nn
from transformers import BertTokenizer
from torch.utils.tensorboard import SummaryWriter

from utils import read_jsonl_data, build_tok_vocab, load_config, load_tokenizer
from models import BiLSTM
from data_loader import get_loader
from trainer import DevignTrainer

import logging.config
import json

config = json.load(open('./logger.json'))
logging.config.dictConfig(config)

logger = logging.getLogger(__name__)


if __name__ == "__main__":
    # 각종 전역 변수
    MODEL_NAME = 'BiLSTM-CAP'
    config_path = r'./config.json'
    [path_train, path_valid, path_test,
     HIDDEN_DIM, EMBED_SIZE, NUM_LSTM_LAYER, BATCH_SIZE,
     n_classes, learning_rate, num_epoch, _tokenizer, SEED, _, block_size,
     output_dir] = load_config(config_path)

    # 실험시 변경 가능한 변수들
    tokenizer = load_tokenizer(_tokenizer)

    # 시드 고정
    random.seed(SEED)
    torch.manual_seed(SEED)

    # CUDA setting 확인
    USE_CUDA = torch.cuda.is_available()
    DEVICE = torch.device("cuda" if USE_CUDA else "cpu")
    logger.info(f'device: {DEVICE}')

    # vocab set
    # data = (project, commit_id, target, func)
    data = read_jsonl_data(path_train)
    functions = [x['func'] for x in data]
    func_tok_vocab_set, func_tok2idx = build_tok_vocab(functions, tokenizer, min_freq=1)
    logger.info(f'Vocab set size: {len(func_tok2idx)}')

    # model
    bilstm_model = BiLSTM(hidden_dim=HIDDEN_DIM,
                          num_lstm_layer=NUM_LSTM_LAYER,
                          n_classes=n_classes,
                          name=MODEL_NAME,
                          embed_size=EMBED_SIZE,
                          vocab_size=len(func_tok2idx))
    logger.info(bilstm_model)
    bilstm_model = bilstm_model.to(DEVICE)
    optimizer = torch.optim.Adam(bilstm_model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    # load data
    train_loader = get_loader(data_path=path_train,
                              batch_size=BATCH_SIZE,
                              tokenizer=tokenizer,
                              tok2idx=func_tok2idx,
                              block_size=block_size)

    eval_loader = get_loader(data_path=path_valid,
                             batch_size=BATCH_SIZE,
                             tokenizer=tokenizer,
                             tok2idx=func_tok2idx,
                             block_size=block_size)

    test_loader = get_loader(data_path=path_test,
                             batch_size=BATCH_SIZE,
                             tokenizer=tokenizer,
                             tok2idx=func_tok2idx,
                             block_size=block_size)

    devign_trainer = DevignTrainer(model=bilstm_model,
                                   optimizer=optimizer,
                                   criterion=criterion,
                                   device=DEVICE,
                                   output_dir=output_dir,
                                   train_loader=train_loader,
                                   eval_loader=eval_loader,
                                   test_loader=test_loader)

    now = datetime.datetime.now().strftime('%Y-%m-%d')
    writer = SummaryWriter(f'logs/{MODEL_NAME}/{now}')

    # train / eval
    for epoch in range(num_epoch):
        print(f'EPOCH: [ {epoch + 1}/{num_epoch} ]')
        train_result, model_summary_path = devign_trainer.train(n_epoch=epoch+1)
        eval_result, _ = devign_trainer.evaluate(mode='eval', n_epoch=epoch+1, model_summary_path=model_summary_path)
        test_result, _ = devign_trainer.evaluate(mode='test', n_epoch=epoch+1, model_summary_path=model_summary_path)

        writer.add_scalars(main_tag='Loss/train_eval',
                           global_step=epoch,
                           tag_scalar_dict={'train_loss': train_result['Avg loss'],
                                            'val_loss': eval_result['Avg loss']})
        writer.add_scalars(main_tag='Acc/train_eval_test',
                           global_step=epoch,
                           tag_scalar_dict={'train_loss': train_result['Avg acc'],
                                            'val_loss': eval_result['Acc'],
                                            'test_loss': test_result['Acc']})
        writer.add_scalars(main_tag='F1/train_eval_test',
                           global_step=epoch,
                           tag_scalar_dict={'train_loss': train_result['Avg f1'],
                                            'val_loss': eval_result['F1'],
                                            'test_loss': test_result['Acc']})

    writer.close()
