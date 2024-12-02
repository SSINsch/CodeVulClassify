import sklearn.metrics as metrics
from sklearn.metrics import f1_score, classification_report
import torch
import os

from utils import to_np, debug_memory
import logging

logger = logging.getLogger(__name__)

class DevignTrainer:
    def __init__(self, model, optimizer, criterion, device,
                 output_dir,
                 train_loader,
                 test_loader=None,
                 eval_loader=None):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.train_loader = train_loader
        self.eval_loader = eval_loader
        self.test_loader = test_loader
        self.total_step = len(train_loader)
        self.device = device
        self.output_dir = output_dir

    def save_model(self, n_epoch, loss, accuracy, f1, mode):
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        model_to_save = self.model.module if hasattr(self.model, 'module') else self.model
        postfix = f'{self.model.name}_{mode}_{n_epoch:03d}.bin'
        output_path = os.path.join(self.output_dir, postfix)

        if mode == 'train':
            torch.save({
                'mode': mode,
                'epoch': n_epoch,
                'model_state_dict': model_to_save.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'loss': loss,
                'accuracy': accuracy,
                'f1-score': f1
            }, output_path)
        else:
            torch.save({
                'mode': mode,
                'epoch': n_epoch,
                'loss': loss,
                'accuracy': accuracy,
                'f1-score': f1
            }, output_path)

        logger.info("Saving model checkpoint to as %s", output_path)

        return output_path

    def load_model(self, model_path):
        checkpoint = torch.load(model_path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        logger.info("Load model checkpoint from %s", model_path)

    def train(self, n_epoch):
        self.model.train()

        total_loss, total_acc, total_f1 = 0, 0, 0
        for step, (_x_func_token, _padded_token_idx_matrix, _y_vuln) in enumerate(self.train_loader):
            logger.debug(f"\t>> Step [{step + 1}/{self.total_step}]\t shape of input: {_padded_token_idx_matrix.shape}")
            # debug_memory()

            # to gpu
            _padded_token_idx_matrix = _padded_token_idx_matrix.to(self.device)
            _y_vuln = _y_vuln.to(self.device)

            # 초기화 해주고
            self.optimizer.zero_grad()

            # to model
            prediction = self.model(_padded_token_idx_matrix)

            # label과 비교해서 loss 구하고 optimizer 처리
            max_predictions, argmax_predictions = prediction.max(1)
            loss = self.criterion(prediction, _y_vuln)
            total_loss = total_loss + loss.item()
            loss.backward()
            self.optimizer.step()

            # torch일 필요가 더 이상 없으므로 np로 만듦
            loss = to_np(loss)
            accuracy = (_y_vuln == argmax_predictions).float().mean()
            accuracy = to_np(accuracy)
            max_predictions = to_np(max_predictions)
            _y_vuln = to_np(_y_vuln)
            argmax_predictions = to_np(argmax_predictions)

            score = f1_score(_y_vuln, argmax_predictions, average='macro')
            total_acc = total_acc + accuracy
            total_f1 = total_f1 + score
            if (step % 10 == 0) or ((step+1) == self.total_step):
                print(f"\t>> Step [{step + 1}/{self.total_step}]", end='')
                print(f"\t Loss: {loss:.4f},", end='')
                print(f"\t Accuracy: {accuracy:.4f}, macro-avg f1: {score:.4f}")

        avg_loss = total_loss / self.total_step
        avg_acc = total_acc / self.total_step
        avg_f1 = total_f1 / self.total_step

        train_result = {'Avg acc': avg_acc, 'Avg f1': avg_f1, 'Avg loss': avg_loss}

        model_summary_path = self.save_model(n_epoch, avg_loss, avg_acc, avg_f1, mode='train')

        return train_result, model_summary_path

    def evaluate(self, model_summary_path, n_epoch, mode='test'):
        # mode check
        if (mode == 'test') and (self.test_loader is not None):
            loader = self.test_loader
        elif (mode == 'eval') and (self.eval_loader is not None):
            loader = self.eval_loader
        else:
            raise ValueError('evaluate mode not found')

        self.load_model(model_summary_path)
        self.model.eval()

        argmax_labels_list, argmax_predictions_list = [], []
        total_loss = 0

        with torch.no_grad():
            for step, (_x_func_token, _padded_token_idx_matrix, _y_vuln) in enumerate(loader):
                # to gpu
                _padded_token_idx_matrix = _padded_token_idx_matrix.to(self.device)
                _y_vuln = _y_vuln.to(self.device)

                # to model
                prediction = self.model(_padded_token_idx_matrix)
                loss = self.criterion(prediction, _y_vuln)
                total_loss = total_loss + loss.item()

                # label과 비교해서 loss 구하고 optimizer 처리
                max_predictions, argmax_predictions = prediction.max(1)
                argmax_labels_list.append(_y_vuln)
                argmax_predictions_list.append(argmax_predictions)

        # Acc
        argmax_labels = torch.cat(argmax_labels_list, 0)
        argmax_predictions = torch.cat(argmax_predictions_list, 0)
        accuracy = (argmax_labels == argmax_predictions).float().mean()
        accuracy = to_np(accuracy)

        # f1 score
        argmax_labels_np_array = to_np(argmax_labels)
        argmax_predictions_np_array = to_np(argmax_predictions)
        macro_f1_score = f1_score(argmax_labels_np_array, argmax_predictions_np_array, average='macro')

        avg_loss = total_loss / self.total_step
        print(f"[ {mode} ] >> ", end='')
        print(f"\t Loss: {avg_loss:.4f},", end='')
        print(f"\t Accuracy: {accuracy:.4f}, macro-avg f1: {macro_f1_score:.4f}")
        # target_names = ['negative', 'positive']
        print(classification_report(argmax_labels_np_array, argmax_predictions_np_array))

        result = {'Acc': accuracy, 'F1': macro_f1_score, 'Avg loss': avg_loss}

        model_summary_path = self.save_model(n_epoch, avg_loss, accuracy, macro_f1_score, mode=mode)

        return result, model_summary_path
