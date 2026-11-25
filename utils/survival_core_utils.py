from argparse import Namespace
import os
from .regularization import Regularization

import numpy as np
from sksurv.metrics import concordance_index_censored

import torch

from datasets.dataset_generic import save_splits
from utils.survival_utils import *
import wandb


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, warmup=5, patience=15, stop_epoch=20, verbose=False):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 20
            stop_epoch (int): Earliest epoch possible for stopping
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
        """
        self.warmup = warmup
        self.patience = patience
        self.stop_epoch = stop_epoch
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf

    def __call__(self, epoch, val_loss, model, ckpt_name = 'checkpoint.pt'):

        score = val_loss
        # score = -val_loss

        if epoch < self.warmup:
            pass
        elif self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, ckpt_name)
        elif score <= self.best_score:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience and epoch > self.stop_epoch:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, ckpt_name)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, ckpt_name):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), ckpt_name)
        self.val_loss_min = val_loss

class EarlyStopping_cindex:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, warmup=5, patience=15, stop_epoch=20, verbose=False):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 20
            stop_epoch (int): Earliest epoch possible for stopping
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
        """
        self.warmup = warmup
        self.patience = patience
        self.stop_epoch = stop_epoch
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf

    def __call__(self, epoch, val_loss, model, ckpt_name = 'checkpoint.pt'):

        score = val_loss
        # score = -val_loss

        if epoch < self.warmup:
            pass
        elif self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, ckpt_name)
        elif score <= self.best_score:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience and epoch > self.stop_epoch:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, ckpt_name)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, ckpt_name):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), ckpt_name)
        self.val_loss_min = val_loss

class Monitor_CIndex:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 20
            stop_epoch (int): Earliest epoch possible for stopping
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
        """
        self.best_score = None

    def __call__(self, val_cindex, model, ckpt_name:str='checkpoint.pt'):

        score = val_cindex

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(model, ckpt_name)
        elif score > self.best_score:
            self.best_score = score
            self.save_checkpoint(model, ckpt_name)
        else:
            pass

    def save_checkpoint(self, model, ckpt_name):
        '''Saves model when validation loss decrease.'''
        torch.save(model.state_dict(), ckpt_name)



def train(datasets: tuple, cur: int, args: Namespace):
    """   
        train for a single fold
    """
    print('\nTraining Fold {}!'.format(cur))
    writer_dir = os.path.join(args.results_dir, str(cur))
    if not os.path.isdir(writer_dir):
        os.mkdir(writer_dir)

    if args.log_data:
        from torch.utils.tensorboard.writer import SummaryWriter
        writer = SummaryWriter(writer_dir, flush_secs=15)
    else:
        writer = None

    if (args.k_fold):
        print('K-fold cross validation')
        train_split, val_split = datasets
    else:
        print('\nInit train/val/test splits...', end=' ')
        train_split, val_split, test_split = datasets
        save_splits(datasets, ['train', 'val', 'test'], os.path.join(args.results_dir, 'splits_{}.csv'.format(cur)))
        print('Done!')
        print("Training on {} samples".format(len(train_split)))
        print("Validating on {} samples".format(len(val_split)))
        print("Testing on {} samples".format(len(test_split)))

        print('\nInit loss function...', end=' ')
    
    if args.bag_loss == 'ce_surv':
        loss_fn = CrossEntropySurvLoss(alpha=args.alpha_surv)
    elif args.bag_loss == 'nll_surv':
        loss_fn = NLLSurvLoss(alpha=args.alpha_surv)
    elif args.bag_loss == 'cox_surv':
        loss_fn = CoxSurvLoss()
    else:
        raise NotImplementedError
    
    if args.reg_type == 'omic':
        reg_fn = l1_reg_all
    elif args.reg_type == 'pathomic':
        reg_fn = l1_reg_modules
    else:
        reg_fn = None

    print('Done!')
    
    print('\nInit Model...', end=' ')
    model_dict = {"dropout": args.drop_out, 'n_classes': args.n_classes}

    ### you can init other models here by adding more conditions

    ### our proposed model
    if args.model_type == 'ours':
        from models.ours import CIGT_Surv
        if args.mode == 'coattn':
            model_dict = {'omic_sizes': args.omic_sizes, 'n_classes': args.n_classes, 'clinical_mode': args.clinical_mode}
            model = CIGT_Surv(**model_dict)
        elif args.mode == 'pathomic':
            model_dict = {'omic_input_dim': args.omic_input_dim, 'n_classes': args.n_classes, 'clinical_mode': args.clinical_mode}
            model = CIGT_Surv(**model_dict)
        else:
            raise NotImplementedError(f'{args.mode} is not implemented ...')
    else:
        raise NotImplementedError(f'{args.model_type} is not implemented ...')
    
    if hasattr(model, "relocate"):
        model.relocate()
    else:
        model = model.to(torch.device('cuda'))
    
    print('Done!')
    print_network(model)

    if args.reg_type == 'L1':
        reg_fn = Regularization(model, args.lambda_reg, p=1)
    else:
        reg_fn = None

    print('\nInit optimizer ...', end=' ')
    optimizer = get_optim(model, args)
    print('Done!')
    
    print('\nInit Loaders...', end=' ')
    train_loader = get_split_loader(train_split, training=True, testing = args.testing, 
        weighted = args.weighted_sample, mode=args.mode, batch_size=args.batch_size, model_type=args.model_type)
    val_loader = get_split_loader(val_split, testing = args.testing, mode=args.mode, batch_size=args.batch_size, model_type=args.model_type)
    if not args.k_fold:
        test_loader = get_split_loader(test_split, testing = args.testing, mode=args.mode, batch_size=args.batch_size, model_type=args.model_type)
    print('Done!')

    data_loader = {'train': train_loader, 'val': val_loader}
    print('\nSetup EarlyStopping...', end=' ')
    if args.early_stopping:
        if args.k_fold:
            early_stopping = EarlyStopping_cindex(warmup=0, patience=10, stop_epoch=20, verbose = True)
        else:
            early_stopping = EarlyStopping(warmup=0, patience=20, stop_epoch=40, verbose = True)
    else:
        early_stopping = None

    print('\nSetup Validation C-Index Monitor...', end=' ')
    monitor_cindex = Monitor_CIndex()
    print('Done!')

    # Ensure each fold gets its own wandb run (prevent mixing folds' logs)
    # try:
        # if wandb.run is not None:
            # wandb.finish()
    # except Exception:
        # pass

    # run_base = getattr(args, 'wandb_name', None) or getattr(args, 'experiment_name', None) or os.path.basename(args.results_dir)
    # dataset_base = args.data_root_dir.split('/')[-1]
    # run_name = f"{dataset_base}_{run_base}_fold_{cur}"

    # get project name (default to 'survival') and optional entity
    # project_name = getattr(args, 'wandb_project', None) or 'survival'
    # entity_name = getattr(args, 'wandb_entity', None)

    # try:
    #     if entity_name:
    #         wandb.init(reinit=True, name=run_name, group=run_name, project=project_name, entity=entity_name, config=vars(args) if isinstance(args, Namespace) else None)
    #     else:
    #         wandb.init(reinit=True, name=run_name, group=run_name, project=project_name, config=vars(args) if isinstance(args, Namespace) else None)
    # except Exception:
    #     # fallback: try init with minimal args if config causes issues
    #     try:
    #         if entity_name:
    #             wandb.init(reinit=True, name=run_name, group=run_name, project=project_name, entity=entity_name)
    #         else:
    #             wandb.init(reinit=True, name=run_name, group=run_name, project=project_name)
    #     except Exception:
    #         pass

    for epoch in range(args.max_epochs):
        if args.model_type in ['ours', 'bilinear', 'concat', 'survpgc']:
            train_ours(epoch, model, train_loader, optimizer, args.n_classes, writer, loss_fn, reg_fn, args.lambda_reg, args.gc, args.alpha_loss, args.mode)
            stop = validate_ours(cur, epoch, model, val_loader, args.n_classes, early_stopping, monitor_cindex, writer, loss_fn, reg_fn, args.lambda_reg, args.results_dir, args.alpha_loss, args.mode)
        else:
            raise NotImplementedError(f'{args.model_type} is not implemented ...')
        if stop:
            break

    print('Done!')
    # if not use early stopping, save the final model
    # torch.save(model.state_dict(), os.path.join(args.results_dir, "s_{}_checkpoint.pt".format(cur)))
    model.load_state_dict(torch.load(os.path.join(args.results_dir, "s_{}_checkpoint.pt".format(cur))))
    if args.model_type in ['ours', 'pathomic', 'concat', 'survpgc']:
        _, val_cindex = summary_ours(model, val_loader, args.n_classes, args.mode)
    else:
        raise NotImplementedError(f'{args.model_type} is not implemented ...')
    print('Val c-Index: {:.4f}'.format(val_cindex))

    # finish this fold's wandb run
    try:
        wandb.finish()
    except Exception:
        pass
    return val_cindex

def train_ours(epoch, model, loader, optimizer, n_classes, writer=None, loss_fn=None, reg_fn=None, lambda_reg=0., gc=16, alpha_loss=0., mode='coattn'):   
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu") 
    model.train()
    train_loss_surv, train_loss = 0., 0.

    print('\n')
    all_risk_scores = np.zeros((len(loader)))
    all_censorships = np.zeros((len(loader)))
    all_event_times = np.zeros((len(loader)))
    if mode == 'coattn':
        for batch_idx, (data_WSI, data_omic1, data_omic2, data_omic3, data_omic4, data_omic5, data_omic6, clinical_prompts, label, event_time, c, slide_id) in enumerate(loader):

            data_WSI = data_WSI.to(device)
            data_omic1 = data_omic1.type(torch.FloatTensor).to(device)
            data_omic2 = data_omic2.type(torch.FloatTensor).to(device)
            data_omic3 = data_omic3.type(torch.FloatTensor).to(device)
            data_omic4 = data_omic4.type(torch.FloatTensor).to(device)
            data_omic5 = data_omic5.type(torch.FloatTensor).to(device)
            data_omic6 = data_omic6.type(torch.FloatTensor).to(device)
            # clinical_prompts = clinical_prompts.type(torch.FloatTensor).to(device)
            label = label.type(torch.LongTensor).to(device)
            c = c.type(torch.FloatTensor).to(device)

            hazards, S, Y_hat, A, mmd_loss  = model(mode=mode, x_path=data_WSI, x_omic1=data_omic1, x_omic2=data_omic2, x_omic3=data_omic3, x_omic4=data_omic4, x_omic5=data_omic5, x_omic6=data_omic6, clinical_prompts=clinical_prompts) # return hazards, S, Y_hat, A_raw, results_dict
            surv_loss = loss_fn(hazards=hazards, S=S, Y=label, c=c)
            if mmd_loss is None:
                mmd_loss = 0
            loss = surv_loss + mmd_loss * alpha_loss
            loss_value = loss.item()
            surv_loss_value = surv_loss.item()

            if reg_fn is None:
                loss_reg = 0
            else:
                loss_reg = reg_fn(model) * lambda_reg

            risk = -torch.sum(S, dim=1).detach().cpu().numpy()
            all_risk_scores[batch_idx] = risk
            all_censorships[batch_idx] = c.item()
            all_event_times[batch_idx] = event_time

            train_loss_surv += surv_loss_value
            train_loss += loss_value + loss_reg

            if (batch_idx + 1) % 100 == 0:
                print('batch {}, loss: {:.4f}, surv_loss: {:.4f}, mmd_loss: {:.4f}, label: {}, event_time: {:.4f}, risk: {:.4f}'.format(batch_idx, train_loss, surv_loss, mmd_loss, label.item(), float(event_time), float(risk)))
            loss = loss / gc + loss_reg
            loss.backward()

            if (batch_idx + 1) % gc == 0: 
                optimizer.step()
                optimizer.zero_grad()
    elif mode == 'pathomic':
        for batch_idx, batch in enumerate(loader):
            data_WSI, data_omic, clinical_prompts, label, event_time, c= batch
            # print(f"Processing case_id: {case_id}")
            data_WSI, data_omic = data_WSI.to(device, non_blocking = True), data_omic.to(device, non_blocking = True)
            label = label.to(device, non_blocking = True)
            c = c.to(device, non_blocking=True)
            hazards, S, Y_hat, _, mmd_loss = model(mode=mode, x_path=data_WSI, x_omic=data_omic, clinical_prompts=clinical_prompts)
            
            surv_loss = loss_fn(hazards=hazards, S=S, Y=label, c=c)
            if mmd_loss is None:
                mmd_loss = 0
            loss = surv_loss + mmd_loss * alpha_loss
            loss_value = loss.item()
            surv_loss_value = surv_loss.item()
            if reg_fn is None:
                loss_reg = 0
            else:
                loss_reg = reg_fn(model)

            risk = -torch.sum(S, dim=1).detach().cpu().numpy()
            all_risk_scores[batch_idx] = risk
            all_censorships[batch_idx] = c.item()
            all_event_times[batch_idx] = event_time

            train_loss_surv += surv_loss_value
            train_loss += loss_value + loss_reg

            if (batch_idx + 1) % 100 == 0:
                print('batch {}, loss: {:.4f}, surv_loss: {:.4f}, mmd_loss: {:.4f}, label: {}, event_time: {:.4f}, risk: {:.4f}'.format(batch_idx, train_loss, surv_loss, mmd_loss, label.item(), float(event_time), float(risk)))
            loss = loss / gc + loss_reg
            loss.backward()

            if (batch_idx + 1) % gc == 0: 
                optimizer.step()
                optimizer.zero_grad()
    else :
        raise ValueError(f"Invalid mode {mode}")
    # calculate loss and error for epoch
    train_loss_surv /= len(loader)
    train_loss /= len(loader)
    c_index = concordance_index_censored((1-all_censorships).astype(bool), all_event_times, all_risk_scores, tied_tol=1e-08)[0]
    print('Epoch: {}, train_loss_surv: {:.4f}, train_loss: {:.4f}, train_c_index: {:.4f}'.format(epoch, train_loss_surv, train_loss, c_index))

    if writer:
        writer.add_scalar('train/loss_surv', train_loss_surv, epoch)
        writer.add_scalar('train/loss', train_loss, epoch)
        writer.add_scalar('train/c_index', c_index, epoch)
    # wandb.log({'train/loss_surv': train_loss_surv, 'train/loss': train_loss, 'train/c_index': c_index, 'epoch': epoch})

def validate_ours(cur, epoch, model, loader, n_classes, early_stopping=None, monitor_cindex=None, writer=None, loss_fn=None, reg_fn=None, lambda_reg=0., results_dir=None, alpha_loss=0., mode='coattn'):
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    val_loss_surv, val_loss = 0., 0.
    all_risk_scores = np.zeros((len(loader)))
    all_censorships = np.zeros((len(loader)))
    all_event_times = np.zeros((len(loader)))
    if mode == 'coattn':
        for batch_idx, (data_WSI, data_omic1, data_omic2, data_omic3, data_omic4, data_omic5, data_omic6, clinical_prompts, label, event_time, c, slide_id) in enumerate(loader):

            data_WSI = data_WSI.to(device)
            data_omic1 = data_omic1.type(torch.FloatTensor).to(device)
            data_omic2 = data_omic2.type(torch.FloatTensor).to(device)
            data_omic3 = data_omic3.type(torch.FloatTensor).to(device)
            data_omic4 = data_omic4.type(torch.FloatTensor).to(device)
            data_omic5 = data_omic5.type(torch.FloatTensor).to(device)
            data_omic6 = data_omic6.type(torch.FloatTensor).to(device)
            # clinical_prompts = clinical_prompts.type(torch.FloatTensor).to(device)
            label = label.type(torch.LongTensor).to(device)
            c = c.type(torch.FloatTensor).to(device)

            with torch.no_grad():
                hazards, S, Y_hat, A, mmd_loss = model(mode=mode, x_path=data_WSI, x_omic1=data_omic1, x_omic2=data_omic2, x_omic3=data_omic3, x_omic4=data_omic4, x_omic5=data_omic5, x_omic6=data_omic6, clinical_prompts=clinical_prompts) # return hazards, S, Y_hat, A_raw, results_dict

            surv_loss = loss_fn(hazards=hazards, S=S, Y=label, c=c, alpha=0)
            surv_loss_value = surv_loss.item()
            if mmd_loss is None:
                mmd_loss = 0
            loss = surv_loss + mmd_loss * alpha_loss
            loss_value = loss.item()

            if reg_fn is None:
                loss_reg = 0
            else:
                loss_reg = reg_fn(model) * lambda_reg

            risk = -torch.sum(S, dim=1).cpu().numpy()
            all_risk_scores[batch_idx] = risk
            all_censorships[batch_idx] = c.cpu().numpy()
            all_event_times[batch_idx] = event_time

            val_loss_surv += surv_loss_value
            val_loss += loss_value + loss_reg
    elif mode == 'pathomic':
        for batch_idx, (data_WSI, data_omic, clinical_prompts, label, event_time, c) in enumerate(loader):
            data_WSI, data_omic = data_WSI.to(device), data_omic.to(device)
            label = label.to(device)
            c = c.to(device)

            with torch.no_grad():
                hazards, S, Y_hat, A, mmd_loss = model(mode=mode, x_path=data_WSI, x_omic=data_omic, clinical_prompts=clinical_prompts)
            surv_loss = loss_fn(hazards=hazards, S=S, Y=label, c=c, alpha=0)
            surv_loss_value = surv_loss.item()
            if mmd_loss is None:
                mmd_loss = 0
            loss = surv_loss + mmd_loss * alpha_loss
            loss_value = loss.item()

            if reg_fn is None:
                loss_reg = 0
            else:
                loss_reg = reg_fn(model) * lambda_reg

            risk = -torch.sum(S, dim=1).cpu().numpy()
            all_risk_scores[batch_idx] = risk
            all_censorships[batch_idx] = c.cpu().numpy()
            all_event_times[batch_idx] = event_time

            val_loss_surv += surv_loss_value
            val_loss += loss_value + loss_reg
    else :
        raise ValueError('mode should be either ours or pathomic')
    val_loss_surv /= len(loader)
    val_loss /= len(loader)
    c_index = concordance_index_censored((1-all_censorships).astype(bool), all_event_times, all_risk_scores, tied_tol=1e-08)[0]
    print('Epoch: {}, val_loss_surv: {:.4f}, val_loss: {:.4f}, val_c_index: {:.4f}'.format(epoch, val_loss_surv, val_loss, c_index))

    if writer:
        writer.add_scalar('val/loss_surv', val_loss_surv, epoch)
        writer.add_scalar('val/loss', val_loss, epoch)
        writer.add_scalar('val/c-index', c_index, epoch)
    # wandb.log({'val/loss_surv': val_loss_surv, 'val/loss': val_loss, 'val/c_index': c_index, 'epoch': epoch})

    if early_stopping:
        assert results_dir
        early_stopping(epoch, c_index, model, ckpt_name=os.path.join(results_dir, "s_{}_checkpoint.pt".format(cur)))
        
        if early_stopping.early_stop:
            print("Early stopping")
            return True

    return False

def summary_ours(model, loader, n_classes, mode='coattn'):
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    test_loss = 0.

    all_risk_scores = np.zeros((len(loader)))
    all_censorships = np.zeros((len(loader)))
    all_event_times = np.zeros((len(loader)))

    slide_ids = loader.dataset.slide_data['slide_id']
    patient_results = {}
    if mode == 'coattn':
        for batch_idx, (data_WSI, data_omic1, data_omic2, data_omic3, data_omic4, data_omic5, data_omic6, clinical_prompts, label, event_time, c, slide_id) in enumerate(loader):
            
            data_WSI = data_WSI.to(device)
            data_omic1 = data_omic1.type(torch.FloatTensor).to(device)
            data_omic2 = data_omic2.type(torch.FloatTensor).to(device)
            data_omic3 = data_omic3.type(torch.FloatTensor).to(device)
            data_omic4 = data_omic4.type(torch.FloatTensor).to(device)
            data_omic5 = data_omic5.type(torch.FloatTensor).to(device)
            data_omic6 = data_omic6.type(torch.FloatTensor).to(device)
            # clinical_prompts = clinical_prompts.type(torch.FloatTensor).to(device)
            label = label.type(torch.LongTensor).to(device)
            c = c.type(torch.FloatTensor).to(device)
            slide_id = slide_ids.iloc[batch_idx]

            with torch.no_grad():
                hazards, survival, Y_hat, A, mmd_loss  = model(mode=mode, x_path=data_WSI, x_omic1=data_omic1, x_omic2=data_omic2, x_omic3=data_omic3, x_omic4=data_omic4, x_omic5=data_omic5, x_omic6=data_omic6, clinical_prompts=clinical_prompts) # return hazards, S, Y_hat, A_raw, results_dict

            risk = -torch.sum(survival, dim=1).cpu().numpy()
            all_risk_scores[batch_idx] = risk
            all_censorships[batch_idx] = c
            all_event_times[batch_idx] = event_time
            patient_results.update({slide_id: {'slide_id': np.array(slide_id), 'risk': risk, 'disc_label': label.item(), 'survival': event_time, 'censorship': c}})
    elif mode == 'pathomic':
        for batch_idx, (data_WSI, data_omic, clinical_prompts, label, event_time, c) in enumerate(loader):
            data_WSI = data_WSI.to(device)
            data_omic = data_omic.type(torch.FloatTensor).to(device)
            label = label.type(torch.LongTensor).to(device)
            c = c.type(torch.FloatTensor).to(device)
            slide_id = slide_ids.iloc[batch_idx]

            with torch.no_grad():
                hazards, survival, Y_hat, A, mmd_loss  = model(mode=mode, x_path=data_WSI, x_omic=data_omic, clinical_prompts=clinical_prompts) # return hazards, S, Y_hat, A_raw, results_dict
            
            risk = -torch.sum(survival, dim=1).cpu().numpy()
            all_risk_scores[batch_idx] = risk
            all_censorships[batch_idx] = c
            all_event_times[batch_idx] = event_time
            patient_results.update({slide_id: {'slide_id': np.array(slide_id), 'risk': risk, 'disc_label': label.item(), 'survival': event_time, 'censorship': c}})
    else:
        raise ValueError('Invalid mode')
    c_index = concordance_index_censored((1-all_censorships).astype(bool), all_event_times, all_risk_scores, tied_tol=1e-08)[0]
    return patient_results, c_index