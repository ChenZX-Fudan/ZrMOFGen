import argparse
import os
import csv
import shutil
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from sklearn import metrics
from torch.autograd import Variable
from torch.utils.data import DataLoader

from cgcnn.data import CIFData
from cgcnn.data import collate_pool
from cgcnn.model import CrystalGraphConvNet

parser = argparse.ArgumentParser(description='Crystal gated neural networks - Ensemble mode')
parser.add_argument('modelpaths', nargs='+', help='paths to the trained models (up to 3 models)')
parser.add_argument('cifpath', help='path to the directory of CIF files.')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('-j', '--workers', default=0, type=int, metavar='N',
                    help='number of data loading workers (default: 0)')
parser.add_argument('--disable-cuda', action='store_true',
                    help='Disable CUDA')
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('-o', '--output', default='ensemble_results.csv', type=str,
                    help='output file name (default: ensemble_results.csv)')
parser.add_argument('-d', '--output-dir', default='.', type=str,
                    help='output directory (default: current directory)')
parser.add_argument('--model-names', nargs='+', default=None,
                    help='custom names for the three models (e.g., --model-names model1 model2 model3)')

args = parser.parse_args(sys.argv[1:])

# 验证模型数量
if len(args.modelpaths) > 3:
    print("Warning: More than 3 models provided. Only the first 3 will be used.")
    args.modelpaths = args.modelpaths[:3]
elif len(args.modelpaths) == 0:
    print("Error: At least one model path is required.")
    sys.exit(1)

print(f"Ensemble mode with {len(args.modelpaths)} models")

# 设置模型名称
if args.model_names is None:
    args.model_names = [f"model_{i+1}" for i in range(len(args.modelpaths))]
else:
    if len(args.model_names) != len(args.modelpaths):
        print(f"Warning: Number of model names ({len(args.model_names)}) doesn't match number of models ({len(args.modelpaths)}). Using default names.")
        args.model_names = [f"model_{i+1}" for i in range(len(args.modelpaths))]

# 加载所有模型
models = []
model_args_list = []
normalizers = []

for i, modelpath in enumerate(args.modelpaths):
    if os.path.isfile(modelpath):
        print(f"=> loading model params '{modelpath}'")
        model_checkpoint = torch.load(modelpath,
                                      map_location=lambda storage, loc: storage)
        model_args = argparse.Namespace(**model_checkpoint['args'])
        print(f"=> loaded model params '{modelpath}'")
        model_args_list.append(model_args)
    else:
        print(f"=> no model params found at '{modelpath}'")
        sys.exit(1)

args.cuda = not args.disable_cuda and torch.cuda.is_available()

# 检查所有模型任务类型是否一致
task_types = [model_args.task for model_args in model_args_list]
if len(set(task_types)) > 1:
    print(f"Warning: Models have different task types: {task_types}. This may cause issues.")
    # 使用第一个模型的任务类型作为主任务类型
    main_task = task_types[0]
else:
    main_task = task_types[0]

if main_task == 'regression':
    best_mae_error = 1e10
else:
    best_mae_error = 0.


def main():
    global args, model_args_list, best_mae_error

    # load data
    dataset = CIFData(args.cifpath)
    collate_fn = collate_pool
    test_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.workers, collate_fn=collate_fn,
                             pin_memory=args.cuda)

    # 加载所有模型
    models = []
    normalizers = []
    
    for i, modelpath in enumerate(args.modelpaths):
        # build model
        structures, _, _ = dataset[0]
        orig_atom_fea_len = structures[0].shape[-1]
        nbr_fea_len = structures[1].shape[-1]
        
        model_args = model_args_list[i]
        model = CrystalGraphConvNet(orig_atom_fea_len, nbr_fea_len,
                                    atom_fea_len=model_args.atom_fea_len,
                                    n_conv=model_args.n_conv,
                                    h_fea_len=model_args.h_fea_len,
                                    n_h=model_args.n_h,
                                    classification=True if model_args.task ==
                                    'classification' else False)
        if args.cuda:
            model.cuda()

        # define loss func
        if model_args.task == 'classification':
            criterion = nn.NLLLoss()
        else:
            criterion = nn.MSELoss()

        normalizer = Normalizer(torch.zeros(3))

        # load model checkpoint
        if os.path.isfile(modelpath):
            print(f"=> loading model '{modelpath}'")
            checkpoint = torch.load(modelpath,
                                    map_location=lambda storage, loc: storage)
            model.load_state_dict(checkpoint['state_dict'])
            normalizer.load_state_dict(checkpoint['normalizer'])
            print(f"=> loaded model '{modelpath}' (epoch {checkpoint['epoch']}, validation {checkpoint['best_mae_error']})")
        else:
            print(f"=> no model found at '{modelpath}'")
            sys.exit(1)
        
        models.append(model)
        normalizers.append(normalizer)

    # 进行ensemble预测
    ensemble_predict(test_loader, models, normalizers, args)


def ensemble_predict(val_loader, models, normalizers, args):
    """Ensemble prediction using multiple models"""
    batch_time = AverageMeter()
    
    # 获取主模型的任务类型
    main_model_idx = 0
    main_task = model_args_list[main_model_idx].task
    
    if main_task == 'regression':
        mae_errors = [AverageMeter() for _ in range(len(models))]
        ensemble_mae_errors = AverageMeter()
    else:
        accuracies = [AverageMeter() for _ in range(len(models))]
        precisions = [AverageMeter() for _ in range(len(models))]
        recalls = [AverageMeter() for _ in range(len(models))]
        fscores = [AverageMeter() for _ in range(len(models))]
        auc_scores = [AverageMeter() for _ in range(len(models))]
        ensemble_accuracies = AverageMeter()
        ensemble_precisions = AverageMeter()
        ensemble_recalls = AverageMeter()
        ensemble_fscores = AverageMeter()
        ensemble_auc_scores = AverageMeter()

    # 初始化测试数据收集
    test_targets = []
    test_cif_ids = []
    test_predictions = [[] for _ in range(len(models))]  # 每个模型的预测
    test_ensemble_predictions = []  # 集成预测

    # switch to evaluate mode
    for model in models:
        model.eval()

    end = time.time()
    
    for i, (input, target, batch_cif_ids) in enumerate(val_loader):
        with torch.no_grad():
            if args.cuda:
                input_var = (Variable(input[0].cuda(non_blocking=True)),
                             Variable(input[1].cuda(non_blocking=True)),
                             input[2].cuda(non_blocking=True),
                             [crys_idx.cuda(non_blocking=True) for crys_idx in input[3]])
            else:
                input_var = (Variable(input[0]),
                             Variable(input[1]),
                             input[2],
                             input[3])
        
        # 收集所有模型的输出
        all_outputs = []
        all_losses = []
        
        for model_idx, (model, normalizer) in enumerate(zip(models, normalizers)):
            if main_task == 'regression':
                target_normed = normalizer.norm(target)
            else:
                target_normed = target.view(-1).long()
            
            with torch.no_grad():
                if args.cuda:
                    target_var = Variable(target_normed.cuda(non_blocking=True))
                else:
                    target_var = Variable(target_normed)

            # compute output
            output = model(*input_var)
            
            if main_task == 'regression':
                criterion = nn.MSELoss()
                loss = criterion(output, target_var)
                mae_error = mae(normalizer.denorm(output.data.cpu()), target)
                mae_errors[model_idx].update(mae_error, target.size(0))
                all_losses.append(loss.data.cpu().item())
                
                # 存储预测值
                pred = normalizer.denorm(output.data.cpu())
                test_predictions[model_idx] += pred.view(-1).tolist()
                all_outputs.append(pred.view(-1, 1))
            else:
                criterion = nn.NLLLoss()
                loss = criterion(output, target_var)
                accuracy, precision, recall, fscore, auc_score = class_eval(output.data.cpu(), target)
                accuracies[model_idx].update(accuracy, target.size(0))
                precisions[model_idx].update(precision, target.size(0))
                recalls[model_idx].update(recall, target.size(0))
                fscores[model_idx].update(fscore, target.size(0))
                auc_scores[model_idx].update(auc_score, target.size(0))
                all_losses.append(loss.data.cpu().item())
                
                # 存储预测概率
                pred_prob = torch.exp(output.data.cpu())
                test_predictions[model_idx] += pred_prob[:, 1].tolist()
                all_outputs.append(pred_prob[:, 1].view(-1, 1))
        
        # 计算ensemble预测（简单平均）
        ensemble_output = torch.mean(torch.cat(all_outputs, dim=1), dim=1)
        
        if main_task == 'regression':
            ensemble_mae = mae(ensemble_output, target)
            ensemble_mae_errors.update(ensemble_mae, target.size(0))
            test_ensemble_predictions += ensemble_output.view(-1).tolist()
        else:
            # 对于分类任务，需要将概率转换为类别
            ensemble_pred_labels = (ensemble_output > 0.5).float()
            ensemble_accuracy, ensemble_precision, ensemble_recall, ensemble_fscore, ensemble_auc = \
                class_eval_from_probs(ensemble_output, target)
            ensemble_accuracies.update(ensemble_accuracy, target.size(0))
            ensemble_precisions.update(ensemble_precision, target.size(0))
            ensemble_recalls.update(ensemble_recall, target.size(0))
            ensemble_fscores.update(ensemble_fscore, target.size(0))
            ensemble_auc_scores.update(ensemble_auc, target.size(0))
            test_ensemble_predictions += ensemble_output.view(-1).tolist()
        
        # 修改这里：移除 if i == 0 的限制，保存所有batch的数据
        test_targets += target.view(-1).tolist()
        test_cif_ids += batch_cif_ids

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print(f'Test: [{i}/{len(val_loader)}]\tTime {batch_time.val:.3f} ({batch_time.avg:.3f})')
            for model_idx, model_name in enumerate(args.model_names):
                if main_task == 'regression':
                    print(f'  {model_name}: Loss {all_losses[model_idx]:.4f}, MAE {mae_errors[model_idx].val:.3f} ({mae_errors[model_idx].avg:.3f})')
                else:
                    print(f'  {model_name}: Loss {all_losses[model_idx]:.4f}, Acc {accuracies[model_idx].val:.3f} ({accuracies[model_idx].avg:.3f})')
            if main_task == 'regression':
                print(f'  Ensemble: MAE {ensemble_mae_errors.val:.3f} ({ensemble_mae_errors.avg:.3f})')
            else:
                print(f'  Ensemble: Acc {ensemble_accuracies.val:.3f} ({ensemble_accuracies.avg:.3f})')

    # 验证数据数量
    print(f"\nTotal samples processed: {len(test_cif_ids)}")
    print(f"Expected samples: {len(val_loader.dataset)}")
    
    # 保存结果
    save_ensemble_results(test_cif_ids, test_targets, test_predictions, 
                         test_ensemble_predictions, args, main_task)
    
    # 打印最终结果
    print('\n' + '='*60)
    print('ENSEMBLE RESULTS SUMMARY')
    print('='*60)
    
    for model_idx, model_name in enumerate(args.model_names):
        if main_task == 'regression':
            print(f'{model_name}: MAE = {mae_errors[model_idx].avg:.4f}')
        else:
            print(f'{model_name}: Accuracy = {accuracies[model_idx].avg:.4f}, AUC = {auc_scores[model_idx].avg:.4f}')
    
    print('-'*60)
    if main_task == 'regression':
        print(f'ENSEMBLE (Average): MAE = {ensemble_mae_errors.avg:.4f}')
    else:
        print(f'ENSEMBLE (Average): Accuracy = {ensemble_accuracies.avg:.4f}, AUC = {ensemble_auc_scores.avg:.4f}')
    print('='*60)
    
    if main_task == 'regression':
        return ensemble_mae_errors.avg
    else:
        return ensemble_auc_scores.avg


def save_ensemble_results(cif_ids, targets, all_predictions, ensemble_predictions, args, task_type):
    """保存ensemble预测结果到CSV文件"""
    output_dir = args.output_dir
    output_file = args.output
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    output_path = os.path.join(output_dir, output_file)
    
    # 创建包含所有模型预测的结果文件
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        
        # 写入表头
        header = ['cif_id', 'target']
        for model_name in args.model_names:
            header.append(f'{model_name}_prediction')
        header.append('ensemble_average')
        writer.writerow(header)
        
        # 写入数据
        for i in range(len(cif_ids)):
            row = [cif_ids[i], targets[i]]
            for model_idx in range(len(args.modelpaths)):
                row.append(all_predictions[model_idx][i])
            row.append(ensemble_predictions[i])
            writer.writerow(row)
    
    print(f"Ensemble results saved to: {output_path}")
    


def class_eval_from_probs(pred_probs, target):
    """从预测概率评估分类性能"""
    pred_probs = pred_probs.numpy()
    target = target.numpy()
    pred_label = (pred_probs > 0.5).astype(int)
    target_label = np.squeeze(target)
    
    precision, recall, fscore, _ = metrics.precision_recall_fscore_support(
        target_label, pred_label, average='binary')
    auc_score = metrics.roc_auc_score(target_label, pred_probs)
    accuracy = metrics.accuracy_score(target_label, pred_label)
    
    return accuracy, precision, recall, fscore, auc_score


class Normalizer(object):
    """Normalize a Tensor and restore it later. """
    def __init__(self, tensor):
        """tensor is taken as a sample to calculate the mean and std"""
        self.mean = torch.mean(tensor)
        self.std = torch.std(tensor)

    def norm(self, tensor):
        return (tensor - self.mean) / self.std

    def denorm(self, normed_tensor):
        return normed_tensor * self.std + self.mean

    def state_dict(self):
        return {'mean': self.mean,
                'std': self.std}

    def load_state_dict(self, state_dict):
        self.mean = state_dict['mean']
        self.std = state_dict['std']


def mae(prediction, target):
    """
    Computes the mean absolute error between prediction and target

    Parameters
    ----------

    prediction: torch.Tensor (N, 1)
    target: torch.Tensor (N, 1)
    """
    return torch.mean(torch.abs(target - prediction))


def class_eval(prediction, target):
    prediction = np.exp(prediction.numpy())
    target = target.numpy()
    pred_label = np.argmax(prediction, axis=1)
    target_label = np.squeeze(target)
    if prediction.shape[1] == 2:
        precision, recall, fscore, _ = metrics.precision_recall_fscore_support(
            target_label, pred_label, average='binary')
        auc_score = metrics.roc_auc_score(target_label, prediction[:, 1])
        accuracy = metrics.accuracy_score(target_label, pred_label)
    else:
        raise NotImplementedError
    return accuracy, precision, recall, fscore, auc_score


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


if __name__ == '__main__':
    main()