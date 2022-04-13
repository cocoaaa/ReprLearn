from pathlib import Path
from reprlearn.visualize.utils import show_timgs
from reprlearn.data.datamodules.kshot_datamodule import KShotDataModule

DATA_ROOT = Path('/data/hayley-old/Tenanbaum2000/data')

def test_kshot_datamodule_one_iter():
    dataset_name = 'cifar100'
    data_root = DATA_ROOT
    k_shot = 4
    n_way = 5
    num_tasks_per_iter = 4

    dm_config = {
        'dataset_name': dataset_name,
        'data_root': data_root,
        'k_shot': k_shot,
        'n_way': n_way,
        'num_tasks_per_iter': num_tasks_per_iter
    }
    dm = KShotDataModule(**dm_config)
    print('Datamodule name: ', dm.name)

    # test datamodule for kshot,nway
    dm.prepare_data()

    # test if datamodule creates correctly woring dataloaders
    train_dl = dm.train_dataloader()

    batch = next(iter(train_dl))
    assert len(batch) == num_tasks_per_iter

    for task_set in batch:
        batch_spt, batch_q = task_set['support'], task_set['query']
        batch_spt_x, batch_spt_y = batch_spt
        batch_q_x, batch_q_y = batch_q

        assert len(batch_spt_x) == dm.k_shot * dm.n_way
        assert len(batch_spt_y) == dm.k_shot * dm.n_way

        assert len(batch_q_x) == dm.k_shot * dm.n_way
        assert len(batch_q_y) == dm.k_shot * dm.n_way

        print("batch_spt_x, y", batch_spt_x.shape, batch_spt_y.shape)
        print("batch_q_x, y", batch_q_x.shape, batch_q_y.shape)


def test_kshot_datamodule_two_iters():
    dataset_name = 'cifar100'
    data_root = DATA_ROOT
    k_shot = 4
    n_way = 5
    num_tasks_per_iter = 4

    dm_config = {
        'dataset_name': dataset_name,
        'data_root': data_root,
        'k_shot': k_shot,
        'n_way': n_way,
        'num_tasks_per_iter': num_tasks_per_iter
    }
    dm = KShotDataModule(**dm_config)
    print('Datamodule name: ', dm.name)

    # test datamodule for kshot,nway
    dm.prepare_data()

    # test if datamodule creates correctly woring dataloaders
    train_dl = dm.train_dataloader()

    # test for multiple iteration over dataloader
    for i, batch in enumerate(train_dl):
        if i == 2:
            break
        assert len(batch) == num_tasks_per_iter

        for task_set in batch:
            batch_spt, batch_q = task_set['support'], task_set['query']
            batch_spt_x, batch_spt_y = batch_spt
            batch_q_x, batch_q_y = batch_q

            assert len(batch_spt_x) == dm.k_shot * dm.n_way
            assert len(batch_spt_y) == dm.k_shot * dm.n_way

            assert len(batch_q_x) == dm.k_shot * dm.n_way
            assert len(batch_q_y) == dm.k_shot * dm.n_way

            print("batch_spt_x, y", batch_spt_x.shape, batch_spt_y.shape)
            print("batch_q_x, y", batch_q_x.shape, batch_q_y.shape)

            if i == 0:
                show_timgs(batch_spt_x, title=f'support {i}')
                show_timgs(batch_q_x, title=f'query {i}')


if __name__ == '__main__':
    test_kshot_datamodule_one_iter()
    test_kshot_datamodule_two_iters()