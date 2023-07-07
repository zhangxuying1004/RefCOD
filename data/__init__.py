'''
the codes for processing and loading data.
create by Xuying Zhang
'''

import torch.utils.data as data


def get_dataloader(data_root, shot, trainsize, batchsize=32, num_workers=8, mode='train'):

    from data.refdataset import R2CObjData as Dataset

    if mode == 'train':
        print('load train data...')
        train_data = Dataset(
            data_root=data_root, 
            mode='train',
            shot=shot,
            image_size=trainsize
        )
        train_loader = data.DataLoader(train_data, batch_size=batchsize, shuffle=True, num_workers=num_workers, pin_memory=True, sampler=None, drop_last=True)
        return train_loader

    elif mode == 'val' or mode == 'test':
        print('laod val data...')
        val_data = Dataset(
            data_root=data_root, 
            mode=mode,
            shot=shot,
            image_size=trainsize
        )

        val_loader = data.DataLoader(val_data, batch_size=1, shuffle=False, num_workers=num_workers, pin_memory=True, sampler=None)
    
        return val_loader

    else:
        raise KeyError('mode {} error!'.format(mode))
