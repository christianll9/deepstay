from data import GL_Dataset, TM_Dataset, ES_Dataset
from model import TransformerModel
import train
import torch
import os
from datetime import datetime

import func_argparse

file_dir = os.path.dirname(os.path.abspath(__file__))


def start(train_data_path:str, name:str="exper_" + datetime.now().isoformat().replace(":", "-"),
    data:str="gl", test_data_path:str=None, class_weights:bool=True,
    pretrained_model_path:str=None, use_trained_decoder:bool=True,
    freeze_encoder:bool=None, save_cp_after_epoch:int=None, d_hid:int=2048,
    nlayers:int=6, emb_size:int=512, nhead:int=8, batch_size:int=64,
    lr:float=5E-5, weight_decay:float=1e-5, interp_sec:float=None,
    weighted_loss:bool=True, weighted_ssl_loss:bool=True, max_epochs:int=80,
    patience:int=100, data_aug:bool=True, dropout:float=0.1,
    val_k:int=0, test_k:int=None, test_frac:float=None, val_frac:float=None,
    λ1:float=None, λ2:float=None, **kw_train_args):

    if λ1 is None:
        if data == "tm": λ1 = 5
        else:            λ1 = 0.2

    if λ2 is None:
        if data == "tm": λ2 = λ1
        else:            λ2 = 50*λ1

    if val_frac is None:
        if   data == "gl": val_frac = 0
        elif data == "es": val_frac = 0.1
        else:              val_frac = 0.2

    if test_frac is None:
        if data == "tm": test_frac = 0.2
        else:            test_frac = 0

    if max_epochs is None:
        if    data == "gl": max_epochs = 80
        elif  data == "es": max_epochs = 400
        else:               max_epochs = 50

    if freeze_encoder is None:
        # default is a freezed encoder, if a pre-trained model is given
        freeze_encoder = pretrained_model_path is not None

    torch.manual_seed(0) #reduce randomness of results
    add_features=["time_diff", "velocity"]

    # Load dataset
    if data == "tm":
        #to reduce memory allocation
        data_filter = lambda df: df.drop(columns=['alt', "osmid", "user", 'amenity', "highway"])
        Dataset = TM_Dataset

    elif data == "gl":
        #to reduce memory allocation
        data_filter = lambda df: df.drop(columns=['alt', "osmid", 'transport', 'amenity', "highway"])
        Dataset = GL_Dataset

    elif data == "es":
        data_filter = None
        Dataset = ES_Dataset

    else:
        raise Exception(f"dataset {data} not found. Available: ['tm', 'gl', 'es']")

    trainval_dataset = Dataset(train_data_path, interp_sec=interp_sec,
        add_features=add_features, data_filter=data_filter)
    test_dataset = Dataset(test_data_path, interp_sec=interp_sec,
            add_features=add_features, data_filter=data_filter) if test_data_path else None


    if test_data_path is None and test_k is not None:
        train_dataset, val_dataset, test_dataset = trainval_dataset.split(val_frac=val_frac,
            generator=torch.Generator().manual_seed(42), val_k=val_k, test_k=test_k, test_frac=test_frac)
    else:
        train_dataset, val_dataset = trainval_dataset.split(val_frac=val_frac,
            generator=torch.Generator().manual_seed(42), val_k=val_k)

    train_norm_vals = trainval_dataset.get_norm_params_of_train_data(train_dataset.indices)
    print("datasets created")

    # Define model
    max_length = trainval_dataset.x.shape[-2]
    in_dim = trainval_dataset.x.shape[-1]
    out_dim = trainval_dataset.y.shape[-1]
    model = TransformerModel(
        in_dim=in_dim, out_dim=out_dim, max_length=max_length,
        tm_classifier=data=="tm", d_hid=d_hid, nlayers=nlayers,
        emb_size=emb_size, nhead=nhead, dropout=dropout
    )

    # Warm start
    if pretrained_model_path is not None:
        state_dict = torch.load(pretrained_model_path)
        if not use_trained_decoder:
            [state_dict.pop(key) for key in list(state_dict.keys()) if "decoder" in key]
        model.load_state_dict(state_dict, strict=use_trained_decoder)

        print("model loaded")
    else:
        model.store_norm_vals(*train_norm_vals)

    trainval_dataset.normalize(model.coords_std, model.feat_mean, model.feat_std)
    try:
        test_dataset.normalize(model.coords_std, model.feat_mean, model.feat_std)
    except AttributeError:
        pass

    if freeze_encoder:
        # makes only sense when pretrained_model_path is not None
        model.freeze_encoder()

    # Start training
    epochs, _, _, best_model_state_dict, _ = train.train(model=model, exec_name=name,
        train_data=train_dataset, val_data=val_dataset, test_data=test_dataset, print_batch_loss=False,
        batch_size=batch_size, lr=lr, weight_decay=weight_decay, patience=patience, data_aug=data_aug,
        weighted_loss=weighted_loss, λ1=λ1, λ2=λ2, class_weights=class_weights, max_epochs=max_epochs,
        save_cp_after_epoch=save_cp_after_epoch, weighted_ssl_loss=weighted_ssl_loss, **kw_train_args)

    model_folder = os.path.join(file_dir, "../trained_models")
    os.makedirs(model_folder, exist_ok=True)
    model_save_path = os.path.join(model_folder, f"{name}_e{epochs}.pt")

    # Save best model
    torch.save(best_model_state_dict, model_save_path)

if __name__ == "__main__":
    func_argparse.single_main(start)