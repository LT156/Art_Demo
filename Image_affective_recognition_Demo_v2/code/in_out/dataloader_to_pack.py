
import torch
import os.path as osp
import multiprocessing as mp
import torchvision.transforms as transforms
import numpy as np 

from PIL import Image

from  Abstract_predict_Demo_v2.code.data_object.WordsPredictDataset import WordsPredictDataset

image_net_mean = [0.485, 0.456, 0.406]
image_net_std = [0.229, 0.224, 0.225]

def max_io_workers():
    """return all/max possible available cpus of machine."""
    return max(mp.cpu_count() - 1, 1)

def image_transformation(img_dim, lanczos=True):
    """simple transformation/pre-processing of image data."""

    if lanczos:
        resample_method = Image.LANCZOS
    else:
        resample_method = Image.BILINEAR

    normalize = transforms.Normalize(mean=image_net_mean, std=image_net_std)
    img_transforms = dict()
    img_transforms['train'] = transforms.Compose([transforms.Resize((img_dim, img_dim), resample_method),
                                                  transforms.ToTensor(),
                                                  normalize])

    # Use same transformations as in train (since no data-augmentation is applied in train)
    img_transforms['test'] = img_transforms['train']
    img_transforms['val'] = img_transforms['train']
    img_transforms['rest'] = img_transforms['train']
    return img_transforms


def dataloader_pack(df, args, drop_thres=None):
    dataloaders = dict()
    datasets = dict()
    img_transforms = image_transformation(args.img_dim, lanczos=args.lanczos)

    if args.num_workers == -1:
        n_workers = max_io_workers()
    else:
        n_workers = args.num_workers

    for split, g in df.groupby('split'):
        g.reset_index(inplace=True, drop=True)
        

        if split == 'train' and drop_thres is not None:
            noise_mask = g['emotion_distribution'].apply(lambda x: max(x) > drop_thres)
            print('Keeping {} of the training data, since for the rest their emotion-maximizer is too low.'.format(noise_mask.mean()))
            g = g[noise_mask]
            g.reset_index(inplace=True, drop=True)
        

        img_files = g.apply(lambda x : osp.join(args.img_dir, x.art_style,  x.painting + '.jpg'), axis=1)
        img_files.name = 'image_files'
        
        obj_files = g.apply(lambda x : osp.join(args.obj_dir, x.art_style,  x.painting + '.npz'), axis=1)
        obj_files.name = 'obj_files'
        
        emotion_his = g.emotion_histogram.apply(lambda x:np.array(x)/sum(x))
        emotion_his.name = 'emotion_his'
        dataset = WordsPredictDataset(img_files, obj_files, g.abs_feature,emotion_his,
                                             img_transform=img_transforms[split])

        datasets[split] = dataset
        b_size = args.batch_size if split=='train' else args.batch_size * 2
        dataloaders[split] = torch.utils.data.DataLoader(dataset=dataset,
                                                         batch_size=b_size,
                                                         shuffle=split=='train',
                                                         num_workers=n_workers)
    return dataloaders, datasets