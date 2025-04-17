from sklearn.model_selection import train_test_split
from time import time, perf_counter


def _parse_expperiment_name(args):
    out_name = [f"_{k}_{v}_" for k, v in args.__dict__.items()]
    return "".join(map(str, out_name))[1:-1]


def experiment(
    learner=None,
    features=None,
    target=None,
    frac=1,
    runs=1,
    resample=1,
    sampler=None,
    model_args=None,
    sampling_args=None,
    metrics=None,
):
    """
    Run an experiment with the given configuration, model, dataset, target, and name.
    """
    sampler_name = sampler.__name__ if sampler else None
    print(
        f"[{sampler_name} ]Running experiment with {learner.__name__} on {features.shape[0]} samples"
    )

    eval_metrics = []
    for sample in range(resample * runs):
        if sample % resample == 0:
            train_feat, test_feat, train_target, test_target = train_test_split(
                features, target, test_size=0.2
            )
            val_feat, test_feat, val_target, test_target = train_test_split(
                test_feat, test_target, test_size=0.5
            )
        # for run in range(runs):
        run = sample % resample
        print(f"Run {(sample) + 1}/{runs*resample} ")
        model = learner(**model_args) if model_args else learner()
        if not sampler:
            print("starting training")
            init_train = perf_counter()
            model.fit(train_feat, train_target)
            end_time = perf_counter()
            print(f"Training time: {end_time - init_train:.2f} seconds")
            test_pred = model.predict(test_feat)
            val_pred = model.predict(val_feat)
            for metric in metrics:
                try:
                    eval_metrics.append(
                        {
                            "model": learner.__name__,
                            "method": sampler_name,
                            "metric": metric.__name__,
                            "frac": 1,
                            "test": metric(test_target, test_pred),
                            "val": metric(val_target, val_pred),
                            "train_time": end_time - init_train,
                            "selection_time": 0,
                        }
                    )
                except:
                    eval_metrics.append(
                        {
                            "model": learner.__name__,
                            "method": sampler_name,
                            "metric": metric.__name__,
                            "frac": frac,
                            "test": metric(test_target, test_pred, average="macro"),
                            "val": metric(val_target, val_pred, average="macro"),
                            "train_time": end_time - init_train,
                            "selection_time": 0,
                        }
                    )
        else:
            k = int(len(train_feat) * frac)
            print(f"Selecting {k/len(train_feat) * 100:.2f}% samples")
            t_, sset = sampler(train_feat, k, **sampling_args)
            print(f"Select time: {t_:.2f} seconds")
            print("starting training")
            init_train = perf_counter()
            model.fit(train_feat[sset], train_target[sset])
            end_time = perf_counter()
            print(f"Training time: {end_time - init_train:.2f} seconds")
            test_pred = model.predict(test_feat)
            val_pred = model.predict(val_feat)
            for metric in metrics:
                try:
                    eval_metrics.append(
                        {
                            "model": learner.__name__,
                            "method": sampler_name,
                            "metric": metric.__name__,
                            "frac": frac,
                            "test": metric(test_target, test_pred),
                            "val": metric(val_target, val_pred),
                            "train_time": end_time - init_train,
                            "selection_time": t_,
                        }
                    )
                except:
                    eval_metrics.append(
                        {
                            "model": learner.__name__,
                            "method": sampler_name,
                            "metric": metric.__name__,
                            "frac": frac,
                            "test": metric(test_target, test_pred, average="macro"),
                            "val": metric(val_target, val_pred, average="macro"),
                            "train_time": end_time - init_train,
                            "selection_time": t_,
                        }
                    )
            del model
    return eval_metrics
