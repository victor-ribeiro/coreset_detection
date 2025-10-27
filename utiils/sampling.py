import asyncio
import functools
import json
import aiofiles


# class Subset:
#     def __init__(self, **kwargs):
#         for k, v in kwargs.items():
#             setattr(self, k, v)


# class ExperimentServer:
#     _consumer = asyncio.Queue(2)


#     def start(self):
#         while not self._consumer.empty():
#             pass
def coroutine(f_):
    @functools.wraps(f_)
    async def inner(*args, **kwargs):
        return f_(*args, **kwargs)

    return inner


async def async_sampling(spl_fn, dataset_config, load_fn, name, run, **sampling_args):
    sampler_name = spl_fn.__name__ if spl_fn else None
    # dataset, _ = load_fn(dataset_config)
    load_fn = coroutine(load_fn)
    dataset, _ = await load_fn(dataset_config)
    frac = sampling_args["K"]
    sampling_args["K"] = int(frac * len(dataset)) if frac < 1 else int(frac)
    print(f"collecting {name} {run}")
    t_, sset = await coroutine(spl_fn)(dataset, **sampling_args)
    print(f"collecting {name} {run}| {t_:.2f} seconds")
    sset = sset.tolist()
    sample = {
        "method": sampler_name,
        "selection_time": t_,
        "coreset": sset,
        "name": name,
    }
    # sample = sample | sampling_args
    sample_id = hash(tuple(sample["coreset"]))
    async with aiofiles.open(f"sampling_{sample_id}.json", "w") as f:
        await f.writelines(json.dumps(sample, indent=True))
    print(f"Sample {sample_id} saved.")
    # return sample


# async def async_select(
#     spl_fn,
#     load_fn,
#     dataset_config,
#     runs=1,
#     sampling_args={},
# ):

#     tasks = [
#         asyncio.create_task(
#             async_sampling(spl_fn, load_fn, dataset_config, run, **sampling_args)
#         )
#         for run in range(runs)
#     ]
#     result = await asyncio.gather(*tasks)
#     task_done = []
#     for run, sample in enumerate(result):
#         sample_id = hash(tuple(sample["coreset"]))
#         async with aiofiles.open(f"sampling_{sample_id}.json", "a") as f:
#             await f.writelines(json.dumps(sample, indent=True))
#         task_done.append((run, sample_id))
#     return task_done


async def async_select(
    spl_fn,
    dataset_config,
    load_fn,
    name,
    runs=1,
    **sampling_args,
):
    consumer = asyncio.Queue(2)
    # consumer = asyncio.Queue(runs)
    # consumer = asyncio.create_task(consumer)
    tasks = [
        asyncio.create_task(
            async_sampling(spl_fn, dataset_config, load_fn, run, name, **sampling_args)
        )
        for run in range(runs)
    ]
    for task in tasks:
        await consumer.put(task)
        # consumer.put_nowait(task)
    while True:

        print(f"Tasks in queue: {consumer.qsize()}")
        task = await consumer.get()
        sample = await task
        consumer.task_done()
        if consumer.empty():
            # consumer.shutdown()
            break
    return asyncio.gather(consumer, *tasks)
    # return asyncio.gather(*tasks)


def collect(spl_fn, load_fn, dataset_config, name, runs=1, **sampling_args):
    loop = asyncio.get_event_loop()
    loop.run_until_complete(
        # asyncio.run(
        async_select(
            spl_fn, dataset_config, load_fn, name=name, runs=runs, **sampling_args
        ),
    )
