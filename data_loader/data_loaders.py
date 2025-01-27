import torch
import torch.utils.data
import numpy as np
from datasets import hw_dataset
from datasets import author_hw_dataset
from datasets import author_rimeslines_dataset
from datasets import author_pageXML_dataset
from datasets import pageXML_dataset
from datasets import author_png_txt_dataset
from datasets import png_txt_dataset
from base import BaseDataLoader



def getDataLoader(config,split):
        data_set_name = config['data_loader']['data_set_name']
        data_dir = config['data_loader']['data_dir']
        batch_size = config['data_loader']['batch_size']
        valid_batch_size = config['validation']['batch_size'] if 'batch_size' in config['validation'] else batch_size

        #copy info from main dataloader to validation (but don't overwrite)
        #helps insure same data
        for k,v in config['data_loader'].items():
            if k not in config['validation']:
                config['validation'][k]=v

        if 'augmentation_params' in config['data_loader']:
            aug_param = config['data_loader']['augmentation_params']
        else:
            aug_param = None
        shuffle = config['data_loader']['shuffle']
        if 'num_workers' in config['data_loader']:
            numDataWorkers = config['data_loader']['num_workers']
        else:
            numDataWorkers = 1
        shuffleValid = config['validation']['shuffle']

        if data_set_name=='HWDataset':
            return withCollate(hw_dataset.HWDataset,hw_dataset.collate,batch_size,valid_batch_size,shuffle,shuffleValid,numDataWorkers,split,data_dir,config)
        elif data_set_name=='PageXMLDataset':
            return withCollate(pageXML_dataset.PageXMLDataset,pageXML_dataset.collate,batch_size,valid_batch_size,shuffle,shuffleValid,numDataWorkers,split,data_dir,config)
        elif data_set_name=='PngTxtDataset':
            return withCollate(png_txt_dataset.PngTxtDataset,png_txt_dataset.collate,batch_size,valid_batch_size,shuffle,shuffleValid,numDataWorkers,split,data_dir,config)
        elif data_set_name=='AuthorHWDataset':
            return withCollate(author_hw_dataset.AuthorHWDataset,author_hw_dataset.collate,batch_size,valid_batch_size,shuffle,shuffleValid,numDataWorkers,split,data_dir,config)
        elif data_set_name=='AuthorRIMESLinesDataset':
            return withCollate(author_rimeslines_dataset.AuthorRIMESLinesDataset,author_rimeslines_dataset.collate,batch_size,valid_batch_size,shuffle,shuffleValid,numDataWorkers,split,data_dir,config)
        elif data_set_name=='AuthorPageXMLDataset':
            return withCollate(author_pageXML_dataset.AuthorPageXMLLinesDataset,author_pageXML_dataset.collate,batch_size,valid_batch_size,shuffle,shuffleValid,numDataWorkers,split,data_dir,config)
        elif data_set_name=='AuthorPngTxtDataset':
            return withCollate(author_png_txt_dataset.AuthorPngTxtDataset, author_png_txt_dataset.collate, batch_size,valid_batch_size,shuffle,shuffleValid,numDataWorkers,split,data_dir,config)
        else:
            print('Error, no dataloader has no set for {}'.format(data_set_name))
            exit(1)



def basic(setObj,batch_size,valid_batch_size,shuffle,shuffleValid,numDataWorkers,split,data_dir,config):
    if split=='train':
        trainData = setObj(dirPath=data_dir, split='train', config=config['data_loader'])
        trainLoader = torch.utils.data.DataLoader(trainData, batch_size=batch_size, shuffle=shuffle, num_workers=numDataWorkers)
        validData = setObj(dirPath=data_dir, split='valid', config=config['validation'])
        validLoader = torch.utils.data.DataLoader(validData, batch_size=valid_batch_size, shuffle=shuffleValid, num_workers=numDataWorkers)
        return trainLoader, validLoader
    elif split=='test':
        testData = setObj(dirPath=data_dir, split='test', config=config['validation'])
        testLoader = torch.utils.data.DataLoader(testData, batch_size=valid_batch_size, shuffle=False, num_workers=numDataWorkers)
    elif split=='merge' or split=='merged' or split=='train-valid' or split=='train+valid':
        trainData = setObj(dirPath=data_dir, split=['train','valid'], config=config['data_loader'])
        trainLoader = torch.utils.data.DataLoader(trainData, batch_size=batch_size, shuffle=shuffle, num_workers=numDataWorkers)
        validData = setObj(dirPath=data_dir, split=['train','valid'], config=config['validation'])
        validLoader = torch.utils.data.DataLoader(validData, batch_size=valid_batch_size, shuffle=shuffleValid, num_workers=numDataWorkers)
        return trainLoader, validLoader
    
def withCollate(setObj,collateFunc,batch_size,valid_batch_size,shuffle,shuffleValid,numDataWorkers,split,data_dir,config):
    if split=='train':
        trainData = setObj(dirPath=data_dir, split='train', config=config['data_loader'])
        trainLoader = torch.utils.data.DataLoader(trainData, batch_size=batch_size, shuffle=shuffle, num_workers=numDataWorkers, collate_fn=collateFunc)
        validData = setObj(dirPath=data_dir, split='valid', config=config['validation'])
        if len(validData)>0:
            validLoader = torch.utils.data.DataLoader(validData, batch_size=valid_batch_size, shuffle=shuffleValid, num_workers=numDataWorkers, collate_fn=collateFunc)
        else:
            validLoader = None
        return trainLoader, validLoader
    elif split=='test':
        testData = setObj(dirPath=data_dir, split='test', config=config['validation'])
        testLoader = torch.utils.data.DataLoader(testData, batch_size=valid_batch_size, shuffle=False, num_workers=numDataWorkers, collate_fn=collateFunc)
        return testLoader, None
    elif split=='merge' or split=='merged' or split=='train-valid' or split=='train+valid':
        trainData = setObj(dirPath=data_dir, split=['train','valid'], config=config['data_loader'])
        trainLoader = torch.utils.data.DataLoader(trainData, batch_size=batch_size, shuffle=shuffle, num_workers=numDataWorkers, collate_fn=collateFunc)
        validData = setObj(dirPath=data_dir, split=['train','valid'], config=config['validation'])
        validLoader = torch.utils.data.DataLoader(validData, batch_size=valid_batch_size, shuffle=shuffleValid, num_workers=numDataWorkers, collate_fn=collateFunc)
        return trainLoader, validLoader
    

