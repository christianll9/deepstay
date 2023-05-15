from algorithms import d_star, cb_smot, kang2004_sr

import metrics
import pickle
import os
import func_argparse
import itertools
import time
import random
import math


sr_functions = {
    # Stay region extraction baselines
    "d_star": d_star,
    "cb_smot": cb_smot,
    "kang2004": kang2004_sr
}

file_dir = os.path.dirname(os.path.abspath(__file__))

def sr_eval(algorithms:str="d_star,cb_smot,kang2004",
    kfold:str=None, iter_per_alg:int=10, output_filepath:str=None):

    algorithm_params = {
        "d_star": {
            "d_max":  [20,   30,  40],
            "q":      {
                "values": [20,   30,  40],
                "type": int
            },
            "t_min":  [100, 135, 170],
            "T_stay": [150, 300, 450]
        },
        "cb_smot": {
            "t_min": [150, 300, 450],
            "a":     [0.5, 0.7, 0.9]
        },
        "kang2004": {
            "d_max": [30,   40,  50],
            "t_min": [150, 300, 450]
        },
    }

    selected_algorithms = algorithms.split(",")

    # modifing upper dict
    for name in list(algorithm_params.keys()):
        if name not in selected_algorithms:
            del algorithm_params[name]

    for name, params in algorithm_params.items():
        for param_name, param_val in params.items():
            if type(param_val) == list:
                algorithm_params[name][param_name] = {
                    "values": param_val,
                    "type": float
                }

    kfolds = [int(k) for k in kfold.split(",")] if kfold is not None else range(5)

    for kfold in kfolds:

        print(f"Loading kfold{kfold} data...")
        with open(os.path.join(file_dir, f'../data/tmp/preprocessed/es_kfold/{kfold}_train.pkl'), "rb") as f:
            gdf_train = pickle.load(f)
        with open(os.path.join(file_dir, f'../data/tmp/preprocessed/es_kfold/{kfold}_test.pkl'), "rb") as f:
            gdf_test = pickle.load(f)
        gdf_train = gdf_train[~gdf_train.geometry.is_empty] # remove missing locations
        gdf_train = gdf_train.to_crs(epsg=32611) # project
        gdf_test = gdf_test[~gdf_test.geometry.is_empty] # remove missing locations
        gdf_test = gdf_test.to_crs(epsg=32611) # project

        random.seed(0, version=2)

        for name, params in algorithm_params.items():

            output_str = f"\nEvaluating {name}...\n"
            total_combinations = math.prod([len(vals["values"]) for vals in params.values()])
            if total_combinations <= iter_per_alg:
                # Grid search
                param_names, param_vals = zip(*params.items())
                param_dicts = [dict(zip(param_names, v)) for v in itertools.product(*[vals["values"] for vals in param_vals])]
            else:
                # Random search
                print("Too much combinations for grid search. (Uniform) random search is used instead.")
                param_lists = {}
                for param_name, param in params.items():
                    if param["type"] == float:
                        param_lists[param_name] = \
                            [random.uniform(min(param["values"]), max(param["values"])) for _ in range(iter_per_alg)]
                    elif param["type"] == int:
                        param_lists[param_name] = \
                            [random.randint(min(param["values"]), max(param["values"])) for _ in range(iter_per_alg)]
                    else:
                        raise Exception(f"unsupported type of parameter {name}['{param_name}']")
                param_dicts = [dict(zip(param_lists,t)) for t in zip(*param_lists.values())]

            output_str += f"Total runs: {len(param_dicts)}\n"
            output_str += "=======================================================================================\n"
            output_str += "idx,kfold,f1,acc,time," + ",".join([param_name for param_name in params.keys()])
            print(output_str)
            if output_filepath is not None:
                with open(output_filepath, 'a', encoding='UTF8') as f:
                    f.write(output_str + "\n")
            total_start = time.time()
            best_f1 = 0
            best_f1_idx = 0
            for idx, param_dict in enumerate(param_dicts):
                start = time.time()
                regions = sr_functions[name](gdf_train, **param_dict)
                time_duration = time.time() - start
                sr_pred = regions > 0
                acc = metrics.stay_accuracy(gdf_train["stay_point"], sr_pred)
                f1  = metrics.nonstay_f1(   gdf_train["stay_point"], sr_pred)
                output_str = f"{idx},{kfold},{f1},{acc},{time_duration}," + ",".join([str(val) for val in param_dict.values()])
                print(output_str)
                if output_filepath is not None:
                    with open(output_filepath, 'a', encoding='UTF8') as f:
                        f.write(output_str + "\n")

                if f1 > best_f1:
                    best_f1 = f1
                    best_f1_idx = idx

            output_str = "=======================================================================================\n"
            output_str += f"Total time: {time.time()-total_start}\n"
            output_str += f"Best F1: {best_f1} | Index: {best_f1_idx} | Params: " + ",".join([str(val) for val in param_dicts[best_f1_idx].values()]) + "\n"

            regions = sr_functions[name](gdf_test, **param_dicts[best_f1_idx])
            sr_pred = regions > 0
            acc       = metrics.stay_accuracy(gdf_test["stay_point"], sr_pred)
            f1        = metrics.nonstay_f1(   gdf_test["stay_point"], sr_pred)
            conf_matr = metrics.stay_conf_matrix(gdf_test["stay_point"], sr_pred)

            output_str += f"Test on best (in terms of F1) params:\nF1  = {f1}\nAcc = {acc}\nConfusion Matrix = {conf_matr}"
            print(output_str)
            if output_filepath is not None:
                with open(output_filepath, 'a', encoding='UTF8') as f:
                    f.write(output_str + "\n")





if __name__ == "__main__":
    func_argparse.single_main(sr_eval)