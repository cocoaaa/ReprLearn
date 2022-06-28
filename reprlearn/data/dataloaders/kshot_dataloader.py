from typing import Optional, Callable, List, Dict
import torch
from reprlearn.data.datasets.kshot_dataset import KShotImageDataset
from reprlearn.data.samplers.kshot_sampler import KShotSampler


class KShotDataLoader:

    def __init__(self,
                 dataset: KShotImageDataset,
                 k_shot: int,
                 n_way: int,
                 batch_size: int,
                 max_iter: int,
                 shuffle: Optional[bool] = True,
                 collate_fn: Optional[Callable] = None,
                 ):
        """Dataloader (as a python iterable) for a batch of task-sets.
        Each batch contains `batch_size` number of task-sets.
        Each task-set is a dictionary of support and query samples.

        Args
        ----
        dataset : KShotImageDataset
          must implements `.sample_task_set` method
        batch_size : int
          number of task-sets in a batch returned at each iteration
        num_iter : int
          number of allowed iterations on this dataloader instance
        shuffle : bool
          if true, shuffle the dataset's unique_classes before (random) sampling
          task_classes for each task_set; in addition, we shuffle data indices
          of a task_set when sampling datapts for the support and query samples
        collate_fn : Optional[Callable]. default: torch.stack

        """
        self.dset = dataset
        self.k_shot = k_shot
        self.n_way = n_way
        self.batch_size = batch_size
        self.max_iter = max_iter
        self.shuffle = shuffle
        self.collate_fn = collate_fn or torch.stack
        self.sampler = KShotSampler()
        self.iter_count = 0

    def __iter__(self):
        return self

    def __next__(self) -> List[Dict[str,torch.Tensor]]:
        self.iter_count += 1
        if self.iter_count > self.max_iter:
            raise StopIteration

        batch = []  # batch of task-sets
        for m in range(self.batch_size):
            task_set, global2local = self.dset.sample_task_set(self.n_way)
            local2global = {v: k for k, v in global2local.items()}
            sample = self.sampler.get_support_and_query(task_set,
                                                   num_per_class=self.k_shot,
                                                   shuffle=self.shuffle,
                                                   collate_fn=self.collate_fn,
                                                   global_id2local_id=global2local)

            batch.append(sample)

            # Test
            # print(f'Taskset {m}: ', task_set.unique_classes)
            # support, query = sample['support'], sample['query']  # support/query: Tuple(batch_x, batch_y)
            # batch_x_spt, batch_y_spt = support
            # batch_x_q, batch_y_q = query

            # todo: print out the global class id of the task_classes of this task-set
            # in order to: confirm they are all randomly sampled set of class-labes
            # of the same size (of n_way)

        return batch