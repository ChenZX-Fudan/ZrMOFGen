import os, argparse
import pandas as pd
import numpy as np
from typing import *
import warnings
import pathlib 
import ray
import sys

warnings.filterwarnings('ignore', category=UserWarning, module="pymatgen.io.cif")

if 'colorama' in sys.modules:
    print("Detected colorama, trying to disable...")
    try:
        import colorama
        colorama.deinit()
    except:
        pass

import torch
from torch_scatter import scatter
from torch.utils.data import DataLoader
from torch_geometric.loader import DataLoader

from train.train_utils import load_state
from train.dist_utils import WandbLogger, init_distributed
from train.data_utils import get_local_rank
from models import CrystalGraphConvNet
from configs import BACKBONES, BACKBONE_KWARGS
from train.data_utils import CIFData, GaussianDistance, AtomCustomJSONInitializer

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, default=None)
    parser.add_argument('--ensemble_names', nargs="*", type=str, default=None)
    parser.add_argument('--model_filename', type=str, default=None, help="GAN Model")
    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument('--gpu', action='store_true')
    parser.add_argument('--gpus', action='store_true')
    parser.add_argument('--silent', action='store_true')
    parser.add_argument('--log', action='store_true')
    parser.add_argument('--plot', action='store_true')
    parser.add_argument('--use-artifacts', action='store_true', help="download model artifacts for loading a model...") 
    parser.add_argument('--which_mode', type=str, help="which mode for script?", default="infer", choices=["infer"]) 

    # data
    parser.add_argument('--dataset', type=str, default="cifdata", choices=["cifdata"])
    parser.add_argument('--data_dir_crystal', type=str, required=True, help="Path to crystal data directory")
    parser.add_argument('--task', type=str, default="homo")
    parser.add_argument('--pin_memory', type=bool, default=False)  # False for prediction
    parser.add_argument('--use_tensors', action="store_true", default=True)
    parser.add_argument('--max_num_nbr', type=int, default=12, help="Maximum number of neighbors")
    parser.add_argument('--radius', type=float, default=8.0, help="Cutoff radius for neighbors")
    parser.add_argument('--truncate_above', type=float, default=None, help="property of Crystal data truncation cutoff...")

    # inference
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--dropnan', action="store_true", help="drop nan smiles... useful for ligand model! during inference!")

    # model
    parser.add_argument('--backbone', type=str, default='cgcnn', choices=["cgcnn"])
    parser.add_argument('--load_ckpt_path', type=str, default="models")
    parser.add_argument('--explain', type=bool, default=False, help="gradient hook for CAM...")

    return parser.parse_args()

def create_full_dataloader(opt):
    """Creat full dataset for prediction"""
    print(f"Creating dataset from: {opt.data_dir_crystal}")
    
    # Creat full dataset
    full_dataset = CIFData(
        root_dir=opt.data_dir_crystal,
        max_num_nbr=opt.max_num_nbr,
        radius=opt.radius,
        truncate_above=opt.truncate_above
    )
    
    print(f"Total samples for prediction: {len(full_dataset)}")
    
    # Creat dataloader
    dataloader = DataLoader(
        full_dataset, 
        batch_size=opt.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=opt.pin_memory
    )

    return dataloader

def call_model(opt, mean, std, logger, model_name=None):
    """Loading model"""
    if model_name:
        name_to_use = model_name
    else:
        name_to_use = opt.name
    
    model = BACKBONES.get(opt.backbone, CrystalGraphConvNet)
    model_kwargs = BACKBONE_KWARGS.get(opt.backbone, {})
    model_kwargs.update({"explain": opt.explain})
    
    if opt.backbone in ["cgcnn"]:
        model_kwargs.update({"mean": mean, "std": std})
        model = model(**model_kwargs)
    
    device = torch.device("cuda" if opt.gpu else "cpu")
    model.to(device)
    model.eval()

    # Loading model weight
    path_and_name = os.path.join(opt.load_ckpt_path, f"{name_to_use}.pth")
    print(f"Loading model from: {path_and_name}")
    
    if not os.path.exists(path_and_name):
        raise FileNotFoundError(f"Model file not found: {path_and_name}")
    
    load_state(model, optimizer=None, scheduler_groups=None, 
               path_and_name=path_and_name, model_only=True, 
               use_artifacts=False, logger=logger, name=None)
    
    if torch.__version__.startswith('2.0'):
        model = torch.compile(model)
        print("PyTorch model has been compiled...")
    
    return model

def infer_for_crystal_ensemble(opt, dataloader, models, mean, std):
    """使用多个模型进行ensemble预测"""
    device = torch.device("cuda" if opt.gpu else "cpu")
    
    # 将所有模型移到设备并设置为评估模式
    for model in models:
        model.to(device)
        model.eval()
    
    df_list = []
    processed_count = 0
    total_batches = len(dataloader)
    
    print(f"Starting ensemble inference, {total_batches} batches to process")
    print(f"Using {len(models)} models for ensemble")
    
    for batch_idx, batch_data in enumerate(dataloader):
        try:
            if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == total_batches:
                print(f"Processing batch {batch_idx + 1}/{total_batches}")
            
            data_batch, data_names = batch_data
            data_batch = data_batch.to(device)
            
            # 标准化输入特征
            standardized_x = (data_batch.x - mean) / std

            all_predictions = []
            all_stds = []
            
            # 对每个模型进行预测
            for i, model in enumerate(models):
                with torch.no_grad():
                    try:
                        output = model(standardized_x, data_batch.edge_attr, 
                                     data_batch.edge_index, data_batch.edge_weight,
                                     data_batch.cif_id, data_batch.batch)
                        
                        if isinstance(output, tuple) and len(output) == 2:
                            # 模型返回 (prediction, std)
                            pred, pred_std = output
                            all_predictions.append(pred)
                            all_stds.append(pred_std)
                        else:
                            # 模型只返回 prediction
                            all_predictions.append(output)
                            
                    except Exception as e:
                        print(f"Model {i+1} prediction error: {e}")
                        # 如果某个模型预测失败，用NaN填充
                        nan_tensor = torch.full((len(data_names),), float('nan'), device=device)
                        all_predictions.append(nan_tensor)
                        if all_stds:  # 如果之前有std，也添加NaN
                            all_stds.append(nan_tensor)
                        continue
            
            if not all_predictions:
                print("No valid predictions from any model")
                continue
            
            # 反标准化预测结果
            final_predictions = []
            for pred in all_predictions:
                final_pred = pred * std + mean
                final_predictions.append(final_pred.cpu().numpy().flatten())
            
            # 计算平均值
            valid_predictions = [p for p in all_predictions if not torch.isnan(p).any()]
            if valid_predictions:
                avg_pred = torch.stack(valid_predictions).mean(dim=0)
                avg_pred = avg_pred * std + mean  # 反标准化
                avg_pred = avg_pred.cpu().numpy().flatten()
            else:
                avg_pred = np.full(len(data_names), float('nan'))
            
            # 创建结果字典
            result_dict = {'name': data_names, 'average': avg_pred}
            
            # 添加每个模型的预测值
            for i, model_name in enumerate(opt.ensemble_names):
                if i < len(final_predictions):
                    result_dict[f'pred_{model_name}'] = final_predictions[i]
                else:
                    result_dict[f'pred_{model_name}'] = np.full(len(data_names), float('nan'))
            
            # 如果有标准差，也添加
            if all_stds and len(all_stds) == len(models):
                for i, model_name in enumerate(opt.ensemble_names):
                    if i < len(all_stds):
                        result_dict[f'std_{model_name}'] = all_stds[i].cpu().numpy().flatten()
                    else:
                        result_dict[f'std_{model_name}'] = np.full(len(data_names), float('nan'))
            
            df_list.append(pd.DataFrame(result_dict))
            processed_count += len(data_names)
            
            if opt.gpu and processed_count % 100 == 0:
                torch.cuda.empty_cache()
                
        except Exception as e:
            print(f"Error processing batch {batch_idx + 1}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    if not df_list:
        print("No data processed successfully")
        # 返回包含所有列的空的DataFrame
        columns = ['name', 'average']
        for model_name in opt.ensemble_names:
            columns.extend([f'pred_{model_name}', f'std_{model_name}'])
        return pd.DataFrame(columns=columns)
    
    df = pd.concat(df_list, axis=0, ignore_index=True)
    
    if opt.dropnan:
        select_nans = np.where(df.name.values == "nan")[0]
        df = df.drop(index=select_nans.tolist()).reset_index(drop=True)
    
    print(f"Successfully processed {len(df)} samples with ensemble")
    return df

def infer_for_crystal_single(opt, dataloader, model, mean, std):
    """单个模型预测"""
    device = torch.device("cuda" if opt.gpu else "cpu")
    model.to(device)
    model.eval()
    
    df_list = []
    processed_count = 0
    total_batches = len(dataloader)
    
    print(f"Starting inference, {total_batches} batches to process")
    
    for batch_idx, batch_data in enumerate(dataloader):
        try:
            if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == total_batches:
                print(f"Processing batch {batch_idx + 1}/{total_batches}")
            
            data_batch, data_names = batch_data
            data_batch = data_batch.to(device)
            
            # 标准化输入特征
            standardized_x = (data_batch.x - mean) / std

            with torch.no_grad():
                output = model(standardized_x, data_batch.edge_attr, 
                             data_batch.edge_index, data_batch.edge_weight,
                             data_batch.cif_id, data_batch.batch)
                
                if isinstance(output, tuple) and len(output) == 2:
                    pred, pred_std = output
                else:
                    pred = output
                    pred_std = None
            
            # 反标准化得到最终预测结果
            final_pred = pred * std + mean
            final_pred = final_pred.cpu().numpy().flatten()
            
            if pred_std is not None:
                pred_std = pred_std.cpu().numpy().flatten()
                df_list.append(pd.DataFrame({
                    'name': data_names,
                    'pred': final_pred,
                    'std': pred_std
                }))
            else:
                df_list.append(pd.DataFrame({
                    'name': data_names,
                    'pred': final_pred
                }))
            
            processed_count += len(data_names)
            
            if opt.gpu and processed_count % 100 == 0:
                torch.cuda.empty_cache()
                
        except Exception as e:
            print(f"Error processing batch {batch_idx + 1}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    if not df_list:
        print("No data processed successfully")
        return pd.DataFrame(columns=["name", "pred", "std"] if opt.ensemble_names else ["name", "pred"])
    
    df = pd.concat(df_list, axis=0, ignore_index=True)
    
    if opt.dropnan:
        select_nans = np.where(df.name.values == "nan")[0]
        df = df.drop(index=select_nans.tolist()).reset_index(drop=True)
    
    print(f"Successfully processed {len(df)} samples")
    return df

def infer(opt):
    """Main prediction"""
    logger = None
    
    print("Loading data...")
    dataloader = create_full_dataloader(opt)
    mean, std = 0.0, 1.0
    print(f"Using normalization: mean={mean:.4f}, std={std:.4f}")    
    try:  

        # Loading model
        if opt.ensemble_names:
            models = []
            for model_name in opt.ensemble_names:
                print(f"Loading model: {model_name}")
                model = call_model(opt, mean, std, logger, model_name=model_name)
                models.append(model)
            print(f"Loaded {len(models)} models for ensemble")
            
            # Ensemble Prediction
            df = infer_for_crystal_ensemble(opt, dataloader, models, mean, std)
        else:
            # Single model prediction
            model = call_model(opt, mean, std, logger)
            df = infer_for_crystal_single(opt, dataloader, model, mean, std)
        
        print(f"Inference completed, obtained {len(df)} results")
        
        # Save results
        output_dir = "results"
        pathlib.Path(output_dir).mkdir(exist_ok=True)
        
        if opt.ensemble_names:
            model_names = "_".join([name.replace(".pth", "") for name in opt.ensemble_names])
            output_path = os.path.join(output_dir, f"ensemble_{model_names}_predictions.csv")
        else:
            output_path = os.path.join(output_dir, f"{opt.name}_predictions.csv")
            
        df.to_csv(output_path, index=False)
        print(f"Results saved to: {output_path}")
        
        return df
        
    except Exception as e:
        print(f"Error during inference: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    warnings.simplefilter("ignore")	
    
    print("Starting inference program...")
    opt = get_parser()
    
    print(f"Mode: {opt.which_mode}")
    print(f"Backbone: {opt.backbone}")
    if opt.ensemble_names:
        print(f"Ensemble models: {opt.ensemble_names}")
    else:
        print(f"Model name: {opt.name}")
    print(f"Model path: {opt.load_ckpt_path}")
    print(f"Data path: {opt.data_dir_crystal}")
    
    if opt.which_mode == "infer":
        result = infer(opt)
        if result is not None:
            print("Inference completed successfully!")
        else:
            print("Inference failed!")
    else:
        print(f"Unsupported mode: {opt.which_mode}")

# python 11_predict.py --which_mode infer --backbone cgcnn --name cgcnn_pub_hmof_0.1 --batch_size 32 --data_dir_crystal ./MOFs/HMOF --gpu
# python 11_predict.py --which_mode infer --backbone cgcnn  --batch_size 32 --data_dir_crystal ./MOFs/HMOF --ensemble_names model1 model2 model3 --gpu