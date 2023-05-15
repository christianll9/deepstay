from model import TransformerModel
from data import rot_traj
from torch.utils.data import Subset
from torch import nn
import torch
from evaluation import accuracy
from os import sys, path
from datetime import datetime
import copy
from torch.utils.tensorboard import SummaryWriter

file_dir = path.dirname(path.abspath(__file__))

def ssl_next_angle_label(x, x_next):
    """
    Calculates angle (in [cos, sin]) between last sequence point
    and its successor point. This is used as a SSL pretext task
    """
    Δx = x_next[:, :2] - x[:, -1, :2]
    d = (Δx[..., 0]**2 + Δx[..., 1]**2).sqrt()
    d = torch.where(d!=0, d, torch.tensor(torch.inf, device=d.device))
    #cosine and sine of angle to next trajectory point
    return Δx/d[:, None]

def log(epoch, name, val, logger=None):
    print("epoch = %4d | %s = %0.4f" % (epoch, name, val))
    if logger is not None: logger.add_scalar(name, val, epoch)


def train(model:TransformerModel, train_data:Subset, val_data:Subset=None,
        test_data:Subset=None, data_aug:bool=True, λ1:float=0.1, λ2:float=0.1,
        patience:int=50, max_epochs:int=1000, batch_size:int=16,
        weighted_loss:bool=True, class_weights:bool=True, print_batch_loss = True,
        lr:float=0.001, weight_decay:float=1E-3, tb_log:bool=True,
        exec_name:str="train_" + datetime.now().isoformat().replace(":", "-"),
        num_workers:int=0 if sys.platform == "darwin" else 4,
        save_cp_after_epoch:int=None, weighted_ssl_loss:bool=False
    ) -> None:
    """
    Training
    weighted_loss:          weights the loss by the amount of true (not interpolated) points \
                            lay within the sequences, if false all sequences are weighted the \
                            same. Keep in mind that purely interpolated sequences are removed \
                            from the dataset during preprocessing (see data.split2np_seqs)
    weighted_ssl_loss:      if weights should be applied on SSL Loss
    data_aug:               data augmentation by rotating the trajectory randomly
    save_cps_after_epochs:  save checkpoints according to epoch numbers
    """

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Device:         {device}")
    print(f"Execution name: {exec_name}")

    if val_data  and len(val_data)  == 0: val_data  = None
    if test_data and len(test_data) == 0: test_data = None

    model.to(device=device)
    _, y_train, _, w_y_train, _ = train_data[:] # (train_data[:][1] -> y)

    if model.tm_classifier:

        if class_weights:
            # total frequency of all class labels
            class_prob = (y_train*w_y_train[...,None]/w_y_train.mean()).mean((0,1)).to(device)
            ce_loss_ = nn.CrossEntropyLoss(reduction='none', weight=1/(class_prob+1E-9))
        else:
            ce_loss_ = nn.CrossEntropyLoss(reduction='none')
        ce_loss = lambda pred, target:\
            ce_loss_(pred.view(-1, target.shape[-1]), target.view(-1, target.shape[-1])).view(-1, target.shape[-2])


    else:
        if class_weights:
            y_train_mean = (y_train*w_y_train[...,None]/w_y_train.mean()).nanmean()
            
            bi_cls_w2 = 1/(1 - y_train_mean)
            bi_cls_w1 = 1/y_train_mean - bi_cls_w2
            bi_cls_w1, bi_cls_w2 = bi_cls_w1.to(device), bi_cls_w2.to(device)
            bce_loss_unweighted = nn.BCELoss(reduction='none')
            bce_loss = lambda pred, target: (bi_cls_w1*target + bi_cls_w2) * bce_loss_unweighted(pred, target)
        else:
            bce_loss = nn.BCELoss(reduction='none')
        ce_loss = lambda pred, target: bce_loss(pred, target)[..., 0]

    
    mse = nn.MSELoss(reduction='none')
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Moreau et al. (2021): Split dataset to (train, val, eval) based on users
    train_ldr = torch.utils.data.DataLoader(train_data, batch_size=batch_size,
        shuffle=True, num_workers=num_workers)
    if val_data:
        val_ldr = torch.utils.data.DataLoader(val_data, batch_size=batch_size,
        num_workers=num_workers)
    if test_data:
        test_ldr = torch.utils.data.DataLoader(test_data, batch_size=batch_size,
        num_workers=num_workers)
    logger = SummaryWriter('logs/Ind_%s'%exec_name) if tb_log else None

    break_counter = 0
    best_epoch = 0
    best_epoch_loss = 1E16
    best_model_state_dict = None

    try:
        for epoch in range(max_epochs):
            epoch_loss = 0.0  # sum of avg loss/item/batch
            epoch_main_loss = 0.0
            epoch_ssl1_loss = 0.0
            epoch_ssl2_loss = 0.0
            batch_counter = 0
            model.train()

            for batch in train_ldr:
                x, y, w_x, w_y, x_next = [elem.to(device) for elem in batch]
                with torch.no_grad():
                    if data_aug:
                        x[..., :2], rand_angle = rot_traj(x[..., :2]) #assumes that 0:2 represent coordinates
                        x_next[:, :2] = rot_traj(x_next[:, None, :2], rand_angle)[0][:, 0]
                    
                    # calculate predicted angle after rotation
                    cos_sin_next = ssl_next_angle_label(x, x_next)

                optimizer.zero_grad()
                outp, pred_cos_sin_next, pred_v_next = model(x)

                if not weighted_loss:
                    w_y = 1

                loss =  (w_y*ce_loss(outp, y)).mean()
                seq_w = w_x.mean(-1) if weighted_ssl_loss else 1
                ssl1_loss = λ1*(seq_w*mse(pred_cos_sin_next, cos_sin_next).mean(-1)).mean()
                ssl2_loss = λ2*(seq_w*mse(pred_v_next, x_next[:,2,None])[...,0]).mean() #velocity
                comb_loss = loss + ssl1_loss + ssl2_loss

                epoch_loss += comb_loss.item()
                epoch_ssl1_loss += ssl1_loss.item()
                epoch_ssl2_loss += ssl2_loss.item()
                epoch_main_loss += loss.item()

                comb_loss.backward()
                optimizer.step()

                # Batch report
                if print_batch_loss:
                    print("epoch = %4d   batch = %4d/%4d   loss = %0.4f" %
                        (epoch, batch_counter, len(train_ldr), comb_loss.item()))
                batch_counter += 1

            # Epoch summary report
            with torch.no_grad():
                model.eval()
                log(epoch, "train/Loss", epoch_main_loss, logger)
                log(epoch, "train/Loss (combined)", epoch_loss, logger)
                log(epoch, "train/Loss (SSL)", epoch_ssl1_loss + epoch_ssl2_loss, logger)
                log(epoch, "train/Loss (SSL 1)", epoch_ssl1_loss, logger)
                log(epoch, "train/Loss (SSL 2)", epoch_ssl2_loss, logger)

                # calculate validation loss
                if val_data:
                    val_epoch_loss = 0.0
                    for x, y, _, w_y, _ in val_ldr:
                        x, y, w_y = x.to(device), y.to(device), w_y.to(device)
                        outp, _, __ = model(x)
                        if not weighted_loss:
                            w_y = 1
                        val_epoch_loss += (w_y*ce_loss(outp, y)).mean().item()
                    log(epoch, "val/Loss", val_epoch_loss, logger)
                if test_data:
                    test_epoch_loss = 0.0
                    for x, y, _, w_y, _ in test_ldr:
                        x, y, w_y = x.to(device), y.to(device), w_y.to(device)
                        outp, _, __ = model(x)
                        if not weighted_loss:
                            w_y = 1
                        test_epoch_loss += (w_y*ce_loss(outp, y)).mean().item()
                    log(epoch, "test/Loss", test_epoch_loss, logger)


                acc = {"train": None, "val":None, "test":None}
                for dataname, data in zip(acc.keys(), [train_data, val_data, test_data]):
                    if data is not None:
                        acc[dataname] = accuracy(model, data, train_data.dataset.comp_fun)
                        log(epoch, f"{dataname}/Accuracy", acc[dataname], logger)
                if print_batch_loss:
                    print("") # new line


            # Early stopping
            # based on validation loss, if val_data is specified
            # elif based on test loss, otherwise based on train loss
            if val_data:
                epoch_loss = val_epoch_loss
            elif test_data:
                epoch_loss = test_epoch_loss
            if  epoch_loss < best_epoch_loss:
                break_counter = 0
                best_epoch_loss = epoch_loss
                best_epoch = epoch
                best_model_state_dict = copy.deepcopy(model.state_dict())
            else:
                break_counter += 1
                if break_counter > patience: break

            # Save checkpoint
            if epoch == save_cp_after_epoch:
                print("save checkpoint")
                torch.save(model.state_dict(), path.join(file_dir, f"../trained_models/{exec_name}_e{epoch+1}.pt"))
    except KeyboardInterrupt:
        pass

    print(f"Done. Trained for {epoch+1} epochs.")
    print(f"Execution name: {exec_name}")
    return best_epoch+1, exec_name, acc, best_model_state_dict, str(device)
