import torch, argparse
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, confusion_matrix
from torch.nn import BCELoss
from torch.optim.lr_scheduler import MultiStepLR
from model import *
from utils import get_logger, metrics_c, set_seed, load_fold_data, load_data

import warnings

warnings.filterwarnings("ignore")
torch.backends.cudnn.benchmark = True


def training(model, smiles_train_loader, optimizer, scheduler, loss_f, device):
    model.train()
    loss_record, record_count = 0., 0.
    correct_predictions, total_samples = 0, 0

    for data in smiles_train_loader:
        data = data.to(device)
        labels = data.y.to(device).squeeze()
        # Forward pass
        output = model(data).squeeze()

        # Compute loss
        loss = loss_f(output, labels)
        loss_record += float(loss.item())
        record_count += 1

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        # nn.utils.clip_grad_value_(model.parameters(), clip_value=2)
        optimizer.step()
        # Compute accuracy
        predictions = (output >= 0.5).float() 
        correct_predictions += (predictions == labels).sum().item()
        total_samples += labels.size(0)
    scheduler.step()
    epoch_loss = loss_record / record_count
    epoch_accuracy = correct_predictions / total_samples
    return epoch_loss, epoch_accuracy


def testing(model, bs_valid_loader, loss_f, metric, device):
    model.eval()  # 切换为评估模式

    loss_record, record_count = 0., 0.

    bs_preds = torch.Tensor([])
    bs_tars = torch.Tensor([])
    with torch.no_grad():
        
        for data_task1 in bs_valid_loader:
            # 获取当前任务的验证 DataLoader
            data_bs = data_task1.to(device)
            labels_bs = data_task1.y.to(device)

            output_bs= model(data_bs)

            loss_bs = loss_f(output_bs.squeeze(), labels_bs.squeeze())

            loss = loss_bs
            loss_record += float(loss.item())
            record_count += 1

            bs_pre = output_bs.detach().cpu()


            bs_preds = torch.cat([bs_preds, bs_pre], 0)
            bs_tars = torch.cat([bs_tars, labels_bs.cpu()], 0)
        
            bs_clas = bs_preds > 0.5
            bs_acc, bs_pre, bs_rec, bs_auc = metric(bs_clas.squeeze().numpy(), bs_preds.squeeze().numpy(),
                                                    bs_tars.squeeze().numpy())
    
        conmatrix = confusion_matrix(bs_tars.squeeze().cpu().numpy(), bs_clas.squeeze().cpu().numpy())
        TN = conmatrix[0, 0]
        FP = conmatrix[0, 1]
        FN = conmatrix[1, 0]
        TP = conmatrix[1, 1]
        SPE = TN / (TN + FP)
        SEN = TP / (TP + FN)
        NPV = TN / (TN + FN)
        PPV = TP / (TP + FP)
        MCC = (TP * TN - FP * FN) / ((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN)) ** 0.5

    epoch_loss = loss_record / record_count
    return epoch_loss, bs_acc, bs_pre, bs_rec, bs_auc, SPE, SEN, NPV, PPV, MCC

def main(task, device, train_epoch, seed, fold, batch_size, logger, lr,
        dropout, fp_type, num_heads, dataset_name, model_name):
    global best_test_acc, bs_best_auc, bs_best_acc, rt_best_auc, rt_best_acc, fhm_best_auc, fhm_best_acc, shm_best_auc, shm_best_acc, bs_best_pre, bs_best_re, rt_best_pre, rt_best_re, fhm_best_pre, fhm_best_re, shm_best_pre, shm_best_re, i
    # dataset = ['Smiles']
    dataset_name = dataset_name
    # dataset = ['FHM']
    logger.info('Dataset: {}  task: {}  train_epoch: {}'.format(dataset_name, task, train_epoch))

    set_seed(seed)
    dataset = load_data(dataset_name, shuffle = True)
    fold_result = [[], [], [], [], [], [], [], []]
    fold_result1 = [[], [], [], [], [], [], [], []]
    if task == 'clas':
        loss_f = BCELoss().to(device)
        # loss_f = CapsuleLoss().to(device)
        metric = metrics_c(accuracy_score, precision_score, recall_score, roc_auc_score)
        for fol in range(fold):
            best_val_acc = 0.
            best_test_acc = 0.
            # model = Multitask(dropout=dropout, device=device, fp_type=fp_type, num_heads=num_heads).to(device)
            # model = Multitask_separated_fp(dropout=dropout, device=device, fp_type=fp_type, num_heads=num_heads).to(device)
            model = Multitask_attn_fp(dropout=dropout, device=device, fp_type=fp_type, num_heads=num_heads).to(device)
            # model = graph_only(dropout=dropout, device=device, fp_type=fp_type, num_heads=num_heads).to(device)
            # model = fp_only(dropout=dropout, device=device, fp_type=fp_type, num_heads=num_heads).to(device)
            optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
            scheduler = MultiStepLR(optimizer, milestones=[50], gamma=0.2)
            smiles_train_loader, smiles_valid_loader, smiles_test_loader = load_fold_data(fol, batch_size, 0, fold, dataset)
            logger.info('Dataset: {}  Fold: {:<4d}'.format(moldata, fol))
            logger.info('Dataset: {}'.format(moldata))

            for i in range(1, train_epoch + 1):
                train_loss, train_acc = training(model, smiles_train_loader,
                                      optimizer, scheduler, loss_f, device)
                valid_loss, bs_acc, bs_pre, bs_rec, bs_auc, SPE1, SEN1, NPV1, PPV1, MCC1 = testing(model, smiles_valid_loader, loss_f, metric, device)
                test_loss, ts_acc, ts_pre, ts_rec, ts_auc, SPE2, SEN2, NPV2, PPV2, MCC2= testing(model, smiles_test_loader, loss_f, metric, device)

                logger.info('Dataset: {}  Epoch: {:<3d}  train_loss: {:.4f} train acc: {:.4f}'.format(dataset_name, i, train_loss, train_acc))
                logger.info('Dataset: {}  Epoch: {:<3d}  valid_loss: {:.4f}'.format(dataset_name, i, valid_loss))
                
                logger.info('Dataset: {}  Epoch: {:<3d}  smiles_valid_auc: {:.4f} smile_valid_acc: {:.4f} '
                            'smiles_valid_re: {:.4f} smiles_valid_pre: {:.4f} best_val_acc: {:.4f}'.format(dataset_name, i, bs_auc, bs_acc, bs_rec, bs_pre, best_val_acc))
                logger.info('Dataset: {}  Epoch: {:<3d}  smiles_test_auc: {:.4f} smile_test_acc: {:.4f} '
                            'smiles_test_re: {:.4f} smiles_test_pre: {:.4f} best_test_acc: {:.4f}'.format(dataset_name, i, ts_auc, ts_acc, ts_rec, ts_pre, best_test_acc))
                logger.info('Dataset: {}  Epoch: {:<3d}  smiles_valid_SPE: {:.4f} smile_valid_SEN: {:.4f} '
                            'smiles_valid_NPV: {:.4f} smiles_valid_PPV: {:.4f} smiles_valid_MCC: {:.4f}'.format(dataset_name, i, SPE1, SEN1, NPV1, PPV1, MCC1))
                logger.info('Dataset: {}  Epoch: {:<3d}  smiles_test_SPE: {:.4f} smile_test_SEN: {:.4f} '
                            'smiles_test_NPV: {:.4f} smiles_test_PPV: {:.4f} smiles_test_MCC: {:.4f}'.format(dataset_name, i, SPE2, SEN2, NPV2, PPV2, MCC2))
                logger.info('---------------------------------------------------------------------------------------')

                if bs_acc > best_val_acc:

                    best_val_acc = bs_acc
                    bs_best_acc = bs_acc
                    bs_best_auc = bs_auc
                    bs_best_pre = bs_pre
                    bs_best_re = bs_rec
                   
                    torch.save(model.state_dict(), 'ckpt/fol{}{}_{}.pkl'.format(fol, 'valid', model_name))
                    print('best valid ckpt saved')
                if ts_acc > best_test_acc:
                    best_test_acc = ts_acc
                    torch.save(model.state_dict(), 'ckpt/fol{}{}_{}.pkl'.format(fol, 'test', model_name))
                    print('best test ckpt saved')
            print('best_val_acc:', best_val_acc)
            fold_result[0].append(bs_best_auc)
            fold_result[1].append(bs_best_acc)
           
            fold_result1[0].append(bs_best_pre)
            fold_result1[1].append(bs_best_re)
            logger.info(
                'Dataset: {} fold:{} best_val_auc: {:.4f} best_val_acc: {:.4f} best_val_re: {:.4f} best_val_pre: {:.4f}'.format(dataset_name, fol, bs_best_auc, bs_best_acc, bs_best_re, bs_best_pre))
            logger.info('---------------------------------------------------------------------------------------\n\n\n\n\n')
        # logger.info('Dataset: {} Fold result: {}'.format(dataset, fold_result, fold_result1))
    return fold_result, fold_result1

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='None')

    parser.add_argument('--mode', default='train', type=str, choices=['train'],
                        help='train, test or hyperparameter_search')
    parser.add_argument('--device', type=str, default='cuda:0', help='Which gpu to use if any (default: cuda:0)')
    parser.add_argument('--batch_size', type=int, default=64, help='Input batch size for training')
    parser.add_argument('--train_epoch', type=int, default=150, help='Number of epochs to train (default: 50)')
    parser.add_argument('--lr', type=float, default=5e-5, help='learning rate')
    parser.add_argument('--fold', type=int, default=5, help='Number of folds for cross validation (default: 3)')
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout ratio')
    parser.add_argument('--seed', type=int,default=1234, help="Seed for splitting the dataset")
    parser.add_argument('--num_heads', type=int, default=4, help='Number of attention heads for transformer')
    parser.add_argument('--dataset_name', type=str, default='hERG')
    args = parser.parse_args()

    device = torch.device(args.device)
    moldata = 'ALL'
    task = 'clas'
    model_name = 'mutitask_attn_fp' #related with name of log

    logf = 'log/{}_{}_{}_{}_{}.log'.format(moldata, task, args.mode, args.dataset_name, model_name)
    logger = get_logger(logf)

    # moldata += task
    fp_type = ['morgan', 'maccs', 'rdit', 'apc2d', 'ecfp']

    if args.mode == 'train':
        logger.info('Training')
        fold_result, fold_result1 = main(task, device, args.train_epoch, args.seed, args.fold,
                                         args.batch_size, logger, args.lr, 
                                          args.dropout, fp_type, args.num_heads, args.dataset_name, model_name)
        print('----------------------------------------------------')
        bs_ava_auc = sum(fold_result[0]) / len(fold_result[0])
        bs_ava_acc = sum(fold_result[1]) / len(fold_result[1])
        bs_ava_pre = sum(fold_result1[0]) / len(fold_result1[0])
        bs_ava_re = sum(fold_result1[1]) / len(fold_result1[1])
        logger.info('-------------------------------------------------------------------------------------------------')
        logger.info('smile_ava_auc:{:.4f}smile_ava_acc:{:.4f} smile_ava_pre:{:.4f} smile_ava_re:{:.4f}'.format(bs_ava_auc, bs_ava_acc,bs_ava_re, bs_ava_pre))