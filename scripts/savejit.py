#!/usr/bin/env python
import os
import time
import argparse
import pandas as pd
import pytorch_lightning as pl
import nugraph as ng
import pynuml
import tqdm
import torch

Data = ng.data.H5DataModule
Model = ng.models.NuGraph2

def configure():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Checkpoint file for trained model')
    parser = Data.add_data_args(parser)
    return parser.parse_args()

def savejit(args):

    print('data path =',args.data_path)
    nudata = Data(args.data_path, batch_size=args.batch_size)

    print('using checkpoint =',args.checkpoint)
    model = Model.load_from_checkpoint(args.checkpoint, map_location='cpu')

    trainer = pl.Trainer(limit_predict_batches=1,
                         logger=False)
    model._trainer = trainer

    script = model.to_torchscript()
    torch.jit.save(script, "model.pt")

    out = trainer.predict(model, dataloaders=nudata.test_dataloader())
    print(out)

if __name__ == '__main__':
    args = configure()
    savejit(args)
