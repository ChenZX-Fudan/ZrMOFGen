import logging
from typing import List, Union

import os, sys, csv
import pathlib
import pandas as pd
roots = pathlib.Path(__file__).parent.parent
sys.path.append(roots) #append top directory

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.modules.loss import _Loss
from torch.nn.parallel import DistributedDataParallel
from torch.optim import Optimizer
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup
# from train.dist_utils import to_cuda, get_local_rank, init_distributed, seed_everything, \
#     using_tensor_cores, increase_l2_fetch_granularity, Logger, WandbLogger
from train.dist_utils import to_cuda, get_local_rank, Logger, WandbLogger
# from transformers import AdamW
# import curtsies.fmtfuncs as cf
import torchmetrics

# https://fairscale.readthedocs.io/en/latest/
# from fairscale.nn.checkpoint.checkpoint_activations import checkpoint_wrapper #anytime
# from fairscale.experimental.nn.offload import OffloadModel #Single-GPU
# from fairscale.optim.adascale import AdaScale #DDP
# from fairscale.nn.data_parallel import ShardedDataParallel as ShardedDDP #Sharding
# from fairscale.optim.oss import OSS #Sharding
import shutil
# from torch.distributed.fsdp import FullyShardedDataParallel, CPUOffload
# from torch.distributed.fsdp.wrap import (
					# default_auto_wrap_policy,
					# enable_wrap,
					# wrap,
					# )

def save_state(model: nn.Module, optimizer: Optimizer, scheduler_groups: "list of schedulers", epoch: int, val_loss: int, path_and_name: Union[pathlib.Path, str]):
    """ Saves model, optimizer and epoch states to path (only once per node) 
    Only local rank=0!"""
    if get_local_rank() == 0:
        scheduler_kwargs = {name : sch.state_dict() for name, sch in zip(["step_scheduler", "epoch_scheduler"], scheduler_groups)}
        state_dict = model.module.state_dict() if isinstance(model, DistributedDataParallel) else model.state_dict()
        checkpoint = {
            'model': state_dict,
            'optimizer': optimizer.state_dict(),
            'epoch': epoch,
            'val_loss': val_loss,
            **scheduler_kwargs
        }

        torch.save(checkpoint, str(path_and_name))
        print(f"Save a model in rank {get_local_rank()}!")

def load_state(model: nn.Module, optimizer: Optimizer, scheduler_groups: "list of schedulers", path_and_name: Union[pathlib.Path, str], model_only=False, use_artifacts=False, logger=None, name=None):
    """ Loads model, optimizer and epoch states from path
    Across multi GPUs
    logger and name kwargs are only needed when use_artifacts is turned on"""
    if use_artifacts and opt.log: 
        model_name = path_and_name.split("/")[-1]
        name = model_name.split(".")[0]
        if get_local_rank() == 0:
            prefix_dir = logger.download_artifacts(f"{name}_model_objects")
            shutil.copy(os.path.join(prefix_dir, model_name), path_and_name + ".artifacts") #copy a file from artifcats dir to save dir! add .adtifacts at the end!
        if dist.is_initialized(): 
            dist.barrier(device_ids=[get_local_rank()]) 
#         path_and_name = os.path.join(prefix_dir, model_name)
        path_and_name += ".artifacts"
	
    ckpt = torch.load(path_and_name, map_location={'cuda:0': f'cuda:{get_local_rank()}'}) #model, optimizer, scheduler
    try:
        if isinstance(model, DistributedDataParallel):
            model.module.load_state_dict(ckpt['model']) #If DDP is saved...
        else:
            model.load_state_dict(ckpt["model"])
        if not model_only:
            optimizer.load_state_dict(ckpt["optimizer"])
            step_scheduler = scheduler_groups[0]
            epoch_scheduler = scheduler_groups[1]
            step_scheduler.load_state_dict(ckpt["step_scheduler"])
            epoch_scheduler.load_state_dict(ckpt["epoch_scheduler"])
            val_loss = ckpt["val_loss"]
            epoch = ckpt["epoch"]
        if model_only:
            epoch = 0
            val_loss = 1e20
    except Exception as e:
        print(e)
        if isinstance(model, DistributedDataParallel):
            model.module.load_state_dict(ckpt) #If DDP is saved...
        else:
            model.load_state_dict(ckpt)
        epoch = 0
        val_loss = 1e20
    finally:
        print(f"Loaded a model from rank {get_local_rank()}!")
    return epoch, val_loss

def single_train(args, model, loader, loss_func, epoch_idx, optimizer, scheduler, grad_scaler, local_rank, logger: WandbLogger, tmetrics):
    #add grad_scaler, local_rank,
    model = model.train()
    losses = []
    path_and_name = os.path.join(args.load_ckpt_path, "{}.pth".format(args.name))
    _loss = 0.
    _loss_metrics = 0.

    pbar = tqdm(enumerate(loader), total=len(loader), unit='batch',
                         desc=f'Training Epoch {epoch_idx}', disable=(args.silent or local_rank != 0))
    for step, packs in pbar:
        if args.gpu and args.use_tensors:
            assert args.backbone in ["cphysnet","cschnet","cgcnn","ctorchmdnet", "calignn", "graph_transformer"], "Wrong data format for a given backbone model!"
            pack, names = packs[0], packs[1]
            atom_fea, nbr_fea, nbr_fea_idx, crystal_atom_idx, batch, dists, targetE = pack.x, pack.edge_attr, pack.edge_index, pack.cif_id, pack.batch, pack.edge_weight, pack.y
            pack =  atom_fea, nbr_fea, nbr_fea_idx, batch, dists, targetE
            # print(atom_fea, targetE)
            atom_fea, nbr_fea, nbr_fea_idx, batch, dists, targetE = to_cuda(pack)
        else:
            print("Significant error in dataloader!")
            break
		
        with torch.cuda.amp.autocast(enabled=args.amp):
            preds = model(atom_fea, nbr_fea, nbr_fea_idx, dists, crystal_atom_idx, batch) 
            loss_mse = loss_func(args, preds, targetE) #get_loss_func
            loss_metrics = tmetrics(preds.view(-1,).detach().cpu(), targetE.view(-1,).detach().cpu()) #LOG energy only!
            
        if args.log:
            logger.log_metrics({'rank0_specific_train_loss_mse': loss_mse.item()})
            logger.log_metrics({'rank0_specific_train_loss_mae': loss_metrics})

        loss = loss_mse
        
        grad_scaler.scale(loss).backward()
        # gradient accumulation
        if (step + 1) % args.accumulate_grad_batches == 0 or (step+ 1) == len(loader):
            if args.gradient_clip:
                grad_scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip)
            grad_scaler.step(optimizer)
            grad_scaler.update()
            model.zero_grad(set_to_none=True)
            #scheduler.step() #stepwise (self.last_epoch is called (as a step) internally)  
#         losses.append(loss)
        _loss += loss.item()
        _loss_metrics += loss_metrics.item()
        #if step % 10 == 0: save_state(model, optimizer, scheduler, epoch_idx, path_and_name) #Deprecated
        pbar.set_postfix(mse_loss=loss.item(), mae_loss=loss_metrics.item() if hasattr(loss_metrics, "item") else loss_metrics)

#     return torch.cat(losses, dim=0).mean() #Not MAE
    return _loss/len(loader), _loss_metrics/len(loader) #mean loss; Not MAE


def single_val(args, model, loader, loss_func, optimizer, scheduler, logger: WandbLogger, tmetrics):
    model = model.eval()
    _loss = 0
    _loss_metrics = 0.

    with torch.inference_mode():  
        pbar = tqdm(enumerate(loader), total=len(loader), unit='batch',
                            desc=f'Validation', disable=(args.silent or get_local_rank() != 0))
        for step, packs in pbar:
            pack, names = packs[0], packs[1]

            if args.gpu and args.use_tensors:
                assert args.backbone in ["cphysnet","cschnet","cgcnn","ctorchmdnet", "calignn", "graph_transformer"], "Wrong data format for a given backbone model!"
                atom_fea, nbr_fea, nbr_fea_idx, crystal_atom_idx, batch, dists, targetE = pack.x, pack.edge_attr, pack.edge_index, pack.cif_id, pack.batch, pack.edge_weight, pack.y
                pack =  atom_fea, nbr_fea, nbr_fea_idx, batch, dists, targetE
                atom_fea, nbr_fea, nbr_fea_idx, batch, dists, targetE = to_cuda(pack)	
            else:
                print("Significant error in dataloader!")
                break
		
            with torch.cuda.amp.autocast(enabled=args.amp):
                preds = model(atom_fea, nbr_fea, nbr_fea_idx, dists, crystal_atom_idx, batch) 
                loss_mse = loss_func(args, preds, targetE) #get_loss_func
                loss_metrics = tmetrics(preds.view(-1,).detach().cpu(), targetE.view(-1,).detach().cpu()) #LOG energy only!

            if args.log:
                logger.log_metrics({'rank0_specific_val_loss_mse': loss_mse.item()})
                logger.log_metrics({'rank0_specific_val_loss_mae': loss_metrics})

            loss = loss_mse
            _loss += loss.item()
            _loss_metrics += loss_metrics.item()

    return _loss/len(loader), _loss_metrics/len(loader) #mean loss; Not MAE
                
def single_test(args, model, loader, loss_func, optimizer, scheduler, logger: WandbLogger, tmetrics):
    model = model.eval()
    _loss = 0
    _loss_metrics = 0.

    with torch.inference_mode():  
        pbar = tqdm(enumerate(loader), total=len(loader), unit='batch',
                            desc=f'Test', disable=(args.silent or get_local_rank() != 0))
        for step, packs in pbar:
            pack, names = packs[0], packs[1]

            if args.gpu and args.use_tensors:
                assert args.backbone in ["cphysnet","cschnet","cgcnn","ctorchmdnet", "calignn", "graph_transformer"], "Wrong data format for a given backbone model!"
                atom_fea, nbr_fea, nbr_fea_idx, crystal_atom_idx, batch, dists, targetE = pack.x, pack.edge_attr, pack.edge_index, pack.cif_id, pack.batch, pack.edge_weight, pack.y
                pack =  atom_fea, nbr_fea, nbr_fea_idx, batch, dists, targetE
                atom_fea, nbr_fea, nbr_fea_idx, batch, dists, targetE = to_cuda(pack)	
            else:
                print("Significant error in dataloader!")
                break
		
            with torch.cuda.amp.autocast(enabled=args.amp):
                preds = model(atom_fea, nbr_fea, nbr_fea_idx, dists, crystal_atom_idx, batch) 
                loss_mse = loss_func(args, preds, targetE) #get_loss_func
                loss_metrics = tmetrics(preds.view(-1,).detach().cpu(), targetE.view(-1,).detach().cpu()) #LOG energy only!

            if args.log:
                logger.log_metrics({'rank0_specific_test_loss_mse': loss_mse.item()})
                logger.log_metrics({'rank0_specific_test_loss_mae': loss_metrics})

            loss = loss_mse
            _loss += loss.item()
            _loss_metrics += loss_metrics.item()

    return _loss/len(loader), _loss_metrics/len(loader) #mean loss; Not MAE
	
def train(model: nn.Module,
          get_loss_func: _Loss,
          train_dataloader: DataLoader,
          val_dataloader: DataLoader,
          test_dataloader: DataLoader,
          logger: Logger,
          args):
    """Includes evaluation and testing as well!"""

    # DDP options
    # Init distributed MUST be called in run() function, which calls this train function!

    device = torch.cuda.current_device()
    model.to(device=device)
    local_rank = get_local_rank()
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    tmetrics = torchmetrics.MeanAbsoluteError()

    # training_log
    if local_rank == 0:
        csv_path = os.path.join(args.load_ckpt_path, "training_log.csv")
        with open(csv_path, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['epoch', 'train_loss', 'train_mae', 'val_loss', 'val_mae', 
                           'test_loss', 'test_mae', 'learning_rate', 'best_val_loss'])
        print(f"📊 Training log will be saved to: {csv_path}")

    # Print training parameter details  
    if local_rank == 0:  
        print(f"\n=== Training Parameter Details ===")  
        print(f"Learning Rate: {args.learning_rate}")  
        print(f"Optimizer: {args.optimizer}")  
        print(f"Weight Decay: {args.weight_decay}")  
        print(f"Total Epochs: {args.epoches}")  
        print(f"Batch Size: {train_dataloader.batch_size}")  
        print(f"Training Samples: {len(train_dataloader.dataset)}")  
        print(f"Validation Samples: {len(val_dataloader.dataset)}")  
        print(f"Test Samples: {len(test_dataloader.dataset)}")  
        print(f"AMP Mixed Precision: {args.amp}")  
        print(f"Gradient Clipping: {args.gradient_clip}")  
    
    # DDP Model
    if dist.is_initialized() and not args.shard:
        model = DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank)
        model._set_static_graph()
        print(f"DDP is enabled {dist.is_initialized()} and this is local rank {local_rank} and sharding is {args.shard}!!")    
        model.train()
        if args.log: logger.start_watching(model) #watch a model!
    elif dist.is_initialized() and args.shard:
        my_auto_wrap_policy = functools.partial(
            default_auto_wrap_policy, min_num_params=100
        )
        torch.cuda.set_device(local_rank)
        model = FSDP(model, fsdp_auto_wrap_policy=my_auto_wrap_policy)

    init_start_event = torch.cuda.Event(enable_timing=True)
    init_end_event = torch.cuda.Event(enable_timing=True)

    # Grad scale
    grad_scaler = torch.cuda.amp.GradScaler(enabled=args.amp)
    
    # Optimizer
    if args.optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, betas=(0.9, 0.999))
    elif args.optimizer == 'lamb':
        optimizer = FusedLAMB(model.parameters(), lr=args.learning_rate, betas=(0.9, 0.999),
                              weight_decay=args.weight_decay)
        base_optimizer = FusedLAMB
        base_optimizer_arguments = dict(lr=args.learning_rate, betas=(0.9, 0.999), weight_decay=args.weight_decay)
    elif args.optimizer == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9,
                                    weight_decay=args.weight_decay)
        base_optimizer = torch.optim.SGD
        base_optimizer_arguments = dict(lr=args.learning_rate, momentum=0.9, weight_decay=args.weight_decay)
    elif args.optimizer == 'torch_adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, betas=(0.9, 0.999),
                              weight_decay=args.weight_decay, eps=1e-8)
        base_optimizer = torch.optim.Adam
        base_optimizer_arguments = dict(lr=args.learning_rate, betas=(0.9, 0.999), weight_decay=args.weight_decay, eps=1e-8)
    elif args.optimizer == 'torch_adamw':
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, betas=(0.9, 0.999),
                              weight_decay=args.weight_decay, eps=1e-8)
        base_optimizer = torch.optim.AdamW
        base_optimizer_arguments = dict(lr=args.learning_rate, betas=(0.9, 0.999), weight_decay=args.weight_decay, eps=1e-8)
    elif args.optimizer == 'torch_sparse_adam':
        optimizer = torch.optim.SparseAdam(model.parameters(), lr=args.learning_rate, betas=(0.9, 0.999),
                              eps=1e-8)
        base_optimizer = torch.optim.SparseAdam
        base_optimizer_arguments = dict(lr=args.learning_rate, betas=(0.9, 0.999), eps=1e-8)

    # SCHEDULER
    total_training_steps = len(train_dataloader) * args.epoches
    warmup_steps = total_training_steps // args.warm_up_split
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_training_steps) #can be used for every step (and epoch if wanted); per training step?
    scheduler_re = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, mode="min", factor=0.9, patience=3) #needs a validation metric to reduce LR (perhaps mainly for epoch-wise eval)
    
    # Model path
    checkpoint_path = os.path.join(args.load_ckpt_path, f"{args.name}_checkpoint.pth")
    best_model_path = os.path.join(args.load_ckpt_path, f"{args.name}_best.pth")
    final_model_path = os.path.join(args.load_ckpt_path, f"{args.name}_final.pth")
    
    scheduler_groups = [scheduler, scheduler_re] #step and epoch schedulers
    epoch_start, best_loss = load_state(model, optimizer, scheduler_groups, checkpoint_path, use_artifacts=args.use_artifacts, logger=logger, name=args.name) if args.resume else (0, 1e20)
    
    best_epoch = epoch_start
    
    # DDP training: Total stats (But still across multi GPUs)
    init_start_event.record()

    for epoch_idx in range(epoch_start, args.epoches):
        if isinstance(train_dataloader.sampler, DistributedSampler):
            train_dataloader.sampler.set_epoch(epoch_idx)
        
        ### TRAINING
        train_epoch = single_train
        # DDP training: Individual stats (i.e. train_epoch; also still across multi GPUs)
        loss, loss_metrics = train_epoch(args, model, train_dataloader, get_loss_func, epoch_idx, optimizer, scheduler, grad_scaler, local_rank,
                           logger, tmetrics) #change to single_train with DDP ;; model is AUTO-updated...

        # ZERO RANK LOGGING
        if dist.is_initialized():
            loss = torch.tensor(loss, dtype=torch.float, device=device)
            loss_metrics = torch.tensor(loss_metrics, dtype=torch.float, device=device)

            torch.distributed.all_reduce(loss, op=torch.distributed.ReduceOp.SUM) #Sum to loss
            loss = (loss / world_size).item()
            logging.info(f'Train loss: {loss}')
            torch.distributed.all_reduce(loss_metrics, op=torch.distributed.ReduceOp.SUM) #Sum to loss
            loss_metrics = (loss_metrics / world_size).item()
            logging.info(f'Train MAE: {loss_metrics}')
    
        if args.log: 
            logger.log_metrics({'ALL_REDUCED_train_loss': loss}, epoch_idx) #zero rank only
            logger.log_metrics({'ALL_REDUCED_train_MAE': loss_metrics}, epoch_idx) #zero rank only
        tmetrics.reset()
        
        ### EVALUATION
        evaluate = single_val
        val_loss, val_mae = evaluate(args, model, val_dataloader, get_loss_func, 
                                         optimizer, scheduler, logger, tmetrics)
        if dist.is_initialized():
            val_loss = torch.tensor(val_loss, dtype=torch.float, device=device)
            val_mae = torch.tensor(loss_metrics, dtype=torch.float, device=device)
            torch.distributed.all_reduce(val_loss, op=torch.distributed.ReduceOp.SUM)
            val_loss = (val_loss / world_size).item()
            torch.distributed.all_reduce(loss_metrics, op=torch.distributed.ReduceOp.SUM) #Sum to loss
            loss_metrics = (loss_metrics / world_size).item()
        if args.log: 
            logger.log_metrics({'ALL_REDUCED_val_loss': val_loss}, epoch_idx)
            logger.log_metrics({'ALL_REDUCED_val_MAE': loss_metrics}, epoch_idx) #zero rank only
        tmetrics.reset()
        
        # Learning Rate Scheduling
        scheduler.step()
        scheduler_re.step(val_loss)  # Adjust learning rate using validation loss

        # Save best model (if current validation loss improves)
        if val_loss < best_loss:
            best_loss = val_loss
            best_epoch = epoch_idx
            
            save_state(model, optimizer, scheduler_groups, epoch_idx, val_loss, best_model_path)
            if args.log: 
                logger.log_artifacts(name=f"{args.name}_best_model", dtype="pytorch_models", path_and_name=best_model_path)
            
            if local_rank == 0:
                print(f"🎯 New best model! Epoch: {epoch_idx}, Val Loss: {val_loss:.6f}")

        # Update checkpoint every epoch (overwriting previous) 
        save_state(model, optimizer, scheduler_groups, epoch_idx, val_loss, checkpoint_path)
        if local_rank == 0:
            print(f"💾 Update checkpoint: Epoch {epoch_idx}, Val Loss: {val_loss:.6f}")
        
        ### TESTING
        test_loss, test_mae = single_test(args, model, test_dataloader, get_loss_func, optimizer, scheduler, logger, tmetrics)
        if dist.is_initialized():
            test_loss = torch.tensor(test_loss, dtype=torch.float, device=device)
            test_mae = torch.tensor(loss_metrics, dtype=torch.float, device=device)
            torch.distributed.all_reduce(test_loss, op=torch.distributed.ReduceOp.SUM)
            test_loss = (test_loss / world_size).item()
            torch.distributed.all_reduce(loss_metrics, op=torch.distributed.ReduceOp.SUM) #Sum to loss
            loss_metrics = (loss_metrics / world_size).item()
        if args.log: 
            logger.log_metrics({'ALL_REDUCED_test_loss': test_loss}, epoch_idx) #zero rank only
            logger.log_metrics({'ALL_REDUCED_test_MAE': loss_metrics}, epoch_idx) #zero rank only
        tmetrics.reset()

        
        # === Save to CSV ===
        if local_rank == 0:
            current_lr = scheduler.get_last_lr()[0] if scheduler.get_last_lr() else args.learning_rate
            
            csv_path = os.path.join(args.load_ckpt_path, "training_log.csv")
            with open(csv_path, mode='a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    epoch_idx,           # epoch
                    loss,                # train_loss  
                    loss_metrics,        # train_mae
                    val_loss,            # val_loss
                    val_mae,             # val_mae
                    test_loss,           # test_loss  
                    test_mae,            # test_mae
                    current_lr,          # learning_rate
                    best_loss            # best_val_loss
                ])
            
            print(f"📝 Epoch {epoch_idx} metrics saved to CSV")
    
        model.train()	
    
    init_end_event.record()

    # Save final model after training completes
    if local_rank == 0:
        save_state(model, optimizer, scheduler_groups, epoch_idx, val_loss, final_model_path)
        
        # Record training information  
        model_info_path = os.path.join(args.load_ckpt_path, "training_info.txt")  
        with open(model_info_path, "w") as f:  
            f.write(f"best_epoch: {best_epoch}\n")  
            f.write(f"best_val_loss: {best_loss:.6f}\n")  
            f.write(f"final_epoch: {epoch_idx}\n")  
            f.write(f"total_epochs: {args.epoches}\n")  
        
        print(f"\n=== Training Completed ===")  
        print(f"✅ Best Model: {best_model_path} (Epoch {best_epoch}, Loss: {best_loss:.6f})")  
        print(f"✅ Final Model: {final_model_path}")  
        print(f"✅ Latest Checkpoint: {checkpoint_path}")  
        print(f"📊 Final Epoch: {epoch_idx}, Final Validation Loss: {val_loss:.6f}")  

    if local_rank == 0:  
        print("Loading best model for evaluation...")  
        
        # Correct map_location usage  
        if torch.cuda.is_available():  
            # Use string or lambda function  
            best_checkpoint = torch.load(best_model_path, map_location=lambda storage, loc: storage.cuda())  
        else:  
            best_checkpoint = torch.load(best_model_path, map_location='cpu')  
        
        # Load best model state  
        if isinstance(model, DistributedDataParallel):  
            model.module.load_state_dict(best_checkpoint['model'])  
        else:  
            model.load_state_dict(best_checkpoint['model'])  
        
        print(f"✅ Evaluation will use best model (Epoch {best_epoch}, Val Loss: {best_loss:.6f})")  
    
    return model  

def evaluate_model(args, model, test_loader, logger=None):  
    """Evaluate model and output test results"""  
    device = torch.device("cuda" if args.gpu else "cpu")  
    model.to(device)  
    model.eval()  
    
    # Initialize metrics  
    from torchmetrics import MeanAbsoluteError, MeanSquaredError, R2Score  
    mae_metric = MeanAbsoluteError()  
    mse_metric = MeanSquaredError()  
    r2_metric = R2Score()  
    
    results = []  
    all_targets = []  # Collect all target values for variance calculation  
    
    with torch.no_grad():  
        for one_data_batch in tqdm(test_loader, desc="Evaluating", disable=args.silent):  
            data_batch = one_data_batch[0]  # Get DATA instance  
            data_names = one_data_batch[1]  # Get CIF names  
            data_batch = data_batch.to(device)  
            
            # Forward pass  
            preds = model(data_batch.x, data_batch.edge_attr, data_batch.edge_index,  
                         data_batch.edge_weight, data_batch.cif_id, data_batch.batch)  
            
            targets = data_batch.y  
            
            # Ensure shape matching [batch_size, 1] -> [batch_size]  
            preds = preds.squeeze(-1) if preds.dim() > 1 else preds  
            targets = targets.squeeze(-1) if targets.dim() > 1 else targets  
            
            # Update metrics  
            mae_metric.update(preds.cpu(), targets.cpu())  
            mse_metric.update(preds.cpu(), targets.cpu())  
            r2_metric.update(preds.cpu(), targets.cpu())  
            
            # Collect target values for variance calculation  
            batch_targets = targets.cpu().numpy()  
            all_targets.extend(batch_targets)  
            
            # Collect results  
            batch_preds = preds.cpu().numpy()  
            
            for i, name in enumerate(data_names):  
                results.append({  
                    'name': name,  
                    'pred': batch_preds[i],  
                    'real': batch_targets[i],  
                    'error': abs(batch_preds[i] - batch_targets[i])  
                })  
    
    # Calculate final metrics  
    mae = mae_metric.compute().item()  
    mse = mse_metric.compute().item()  
    r2 = r2_metric.compute().item()  
    rmse = torch.sqrt(torch.tensor(mse)).item()  
    
    # Calculate data variance (target variable variance)  
    all_targets_array = np.array(all_targets)  
    data_variance = np.var(all_targets_array)  
    data_std = np.std(all_targets_array)  
    
    # Output results  
    print(f"\n=== Model Evaluation Results ===")  
    print(f"MAE: {mae:.4f}")  
    print(f"MSE: {mse:.4f}")  
    print(f"RMSE: {rmse:.4f}")  
    print(f"R²: {r2:.4f}")  
    print(f"Data Variance: {data_variance:.4f}")  
    print(f"Data Standard Deviation: {data_std:.4f}")  
    print(f"MSE/Data Variance: {mse/data_variance:.4f}")  # Relative variance ratio  
    
    # Save results to CSV  
    results_df = pd.DataFrame(results)  
    
    # Ensure directory exists  
    os.makedirs(args.load_ckpt_path, exist_ok=True)  
    results_path = os.path.join(args.load_ckpt_path, "test_results.csv")  
    results_df.to_csv(results_path, index=False)  
    print(f"Test results saved to {results_path}")  
    
    # Log to logger (if enabled) 
    if logger and hasattr(args, 'log') and args.log:
        logger.log_metrics({
            'test_mae': mae,
            'test_mse': mse, 
            'test_rmse': rmse,
            'test_r2': r2,
            'data_variance': data_variance,
            'data_std': data_std,
            'mse_variance_ratio': mse/data_variance
        })
    
    return {
        'mae': mae,
        'mse': mse,
        'rmse': rmse,
        'r2': r2,
        'data_variance': data_variance,
        'data_std': data_std,
        'mse_variance_ratio': mse/data_variance,
        'results_df': results_df
    }
