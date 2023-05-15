from typing import Callable
from torch.utils.data import Dataset, DataLoader
from model import TransformerModel
from data import GL_Dataset, ES_Dataset, TM_Dataset
import numpy as np
import pickle
import func_argparse
import torch


def _predict_incomplete_batch(model, x, model_call:Callable=None, y_last_dim:int=1):
    """
    returns detached predictions for x even when not some sequences
    in batch are not complete (indicated with nan)
    """
    if model_call is None:
        model_call = model
    if torch.isnan(x).any():
        y_pred = torch.full([*x.shape[:2], y_last_dim], torch.nan)
        compl_seqs = ~torch.isnan(x).any(1).any(1)
        idxs_incompl = (~compl_seqs).nonzero()
        x_compl_seqs = x[compl_seqs]
        x_incompl_seqs = x[~compl_seqs]
        y_pred_compl_seqs, _, _ = model_call(x_compl_seqs.to(device=model.device))
        y_pred[compl_seqs] = y_pred_compl_seqs.cpu().detach()
        for idx, x_partial_seq in zip(idxs_incompl, x_incompl_seqs):
            y_pred_part_seq_nonan, _, _ = model_call(x_partial_seq[~torch.isnan(x_partial_seq).any(1)][None,...]\
                .to(device=model.device))
            y_pred_part_seq_nonan = y_pred_part_seq_nonan[0].cpu().detach()
            y_pred[idx, :len(y_pred_part_seq_nonan)] = y_pred_part_seq_nonan
    else:
        outp, _, _ = model_call(x.to(device=model.device))
        y_pred = outp.cpu().detach()

    return y_pred


def accuracy(model:TransformerModel, test_set:Dataset, comp_fun:Callable, batch_size:int=128) -> float:

    from_training = model.training
    model.eval()

    if len(test_set) == 0:
        return torch.nan
    accs = []
    batch_weights = []
    device = model.device
    test_dl = DataLoader(test_set, batch_size=batch_size)
    for x, y, _, w_y, _ in test_dl:
        x, y, w_y = x.to(device=device), y.to(device=device), w_y.to(device=device)

        outp, _, __ = model(x)
        
        w_batch = w_y.sum().item()
        if w_batch > 0:
            acc = ((w_y*comp_fun(outp, y)).sum()/w_y.sum()).item()
        else:
            acc = 0
        accs.append(acc)
        batch_weights.append(w_batch)

    if from_training:
        model.train() #return to training mode
    if sum(batch_weights) > 0:
        return np.average(accs, weights=batch_weights)
    else:
        return torch.nan


def get_model_prediction(testdata_filepath:str, model_path:str, data:str="gl",
    batch_size:int=128, dataset_kwargs:dict={}, dataset_interp_sec:int=None,
    tm_test_frac:float=1, tm_test_k:int=0,
    output_filepath:str=None, remove_weight0:bool=True)->tuple[np.ndarray]:
    "Returns numpy vectors for prediction, ground_truth, point weight (not constant if interpolation is used) "

    if "split_kwargs" not in dataset_kwargs:
        dataset_kwargs["split_kwargs"] = {}
    if "include_remain" not in dataset_kwargs["split_kwargs"]:
        dataset_kwargs["split_kwargs"]["include_remain"] = True
    if "y_interp_meth" not in dataset_kwargs["split_kwargs"]:
        dataset_kwargs["split_kwargs"]["y_interp_meth"] = "nearest"
        

    if data == "gl":
        data_filter = lambda df: df.drop(columns=['alt', "osmid", 'transport', 'amenity', "highway"])
        dataset = GL_Dataset(testdata_filepath, interp_sec=dataset_interp_sec,
            data_filter=data_filter, **dataset_kwargs)
        test_dataset = dataset
    elif data == "es":
        dataset = ES_Dataset(testdata_filepath, interp_sec=dataset_interp_sec,
            **dataset_kwargs)
        test_dataset = dataset
    elif data == "tm":
        data_filter = lambda df: df.drop(columns=['alt', "osmid", "user", 'amenity', "highway"])
        dataset = TM_Dataset(testdata_filepath, interp_sec=dataset_interp_sec,
            data_filter=data_filter, **dataset_kwargs)
        # random split independent of users
        _, _, test_dataset = dataset.split(test_frac=tm_test_frac,
            generator=torch.Generator().manual_seed(42), test_k=tm_test_k)

    else:
        raise Exception(f"No implementation for data={data}")

    print("dataset loaded")

    model = TransformerModel(
        in_dim=dataset.x.shape[-1], out_dim=dataset.y.shape[-1], max_length=dataset.x.shape[-2],
        tm_classifier=False, d_hid=2048, nlayers=6, emb_size=512, nhead=8, dropout=0.1
    )
    if data in ["gl", "es"]:
        model_call = model
        gt_call = lambda y: y
    elif data == "tm":
        model_call = lambda x: (model(x)[0].softmax(dim=-1).argmax(dim=-1)[..., None].to(torch.float32), None, None)
        gt_call = lambda y: y.argmax(dim=-1)

    if torch.cuda.is_available():
        state_dict = torch.load(model_path)
    else:
        state_dict = torch.load(model_path, map_location=torch.device('cpu'))
    model.load_state_dict(state_dict)
    del state_dict
    dataset.normalize(model.coords_std, model.feat_mean, model.feat_std)

    model.eval()

    test_dl = DataLoader(test_dataset, batch_size=batch_size)
    pairs = []
    for x, y, _, w_y, _ in test_dl:
        y_pred = _predict_incomplete_batch(model, x, model_call=model_call)
        y = gt_call(y)
        y_pred, y, w_y = [tens.view(-1) for tens in [y_pred, y, w_y]]

        if remove_weight0:
            pred_gt_pairs = [tens[w_y>0] for tens in [y_pred, y, w_y]]
        else:
            pred_gt_pairs = [y_pred, y, w_y]
        pairs.append(pred_gt_pairs)
    pairs = [torch.cat(ls).numpy() for ls in zip(*pairs)]

    if output_filepath is not None:
        with open(output_filepath, "wb") as f:
            pickle.dump(pairs, f)
    
    return pairs


if __name__ == "__main__":
    func_argparse.main()