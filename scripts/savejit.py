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
    #parser.add_argument('--outfile', type=str, required=True,
    #                    help='Output file name (full path)')
    parser = Data.add_data_args(parser)
    return parser.parse_args()

def savejit(args):

    print('data path =',args.data_path)
    nudata = Data(args.data_path, batch_size=args.batch_size)

    print('using checkpoint =',args.checkpoint)
    model = Model.load_from_checkpoint(args.checkpoint, map_location='cpu')

    #accelerator, devices = ng.util.configure_device()
    trainer = pl.Trainer(#accelerator=accelerator, devices=devices,
                         limit_predict_batches=1,
                         logger=False)
    model._trainer = trainer
    #model._trainer = pl.Trainer()

    #compiled_model = torch.compile(model)
    #script = torch.jit.script(compiled_model) 
    script = model.to_torchscript()
    torch.jit.save(script, "model.pt")

    
    ##print('output file =',args.outfile)
    ##if os.path.isfile(args.outfile):
    ##    raise Exception(f'file {args.outfile} already exists!')

    ###script = model.to_torchscript()
    ##script = torch.jit.script(model)
    ###print(script)
    ##torch.jit.save(script, "model.pt")
    
    ## Get a sample input from the dataloader
    #sample_input = next(iter(nudata.test_dataloader())).get_example(0)
    #print(sample_input)

    #traced_model = torch.jit.trace(model, (sample_input.x_dict, sample_input.edge_index_dict))
    ##traced_model = model.to_torchscript(method="trace", sample_input)
    
    ## Save the traced model
    #torch.jit.save(traced_model, "traced_model.pt")

    
    #plot = pynuml.plot.GraphPlot(planes=nudata.planes,
    #                             classes=nudata.semantic_classes)

    #start = time.time()
    out = trainer.predict(model, dataloaders=nudata.test_dataloader())
    print(out)
    #end = time.time()
    #itime = end - start
    #ngraphs = len(nudata.test_dataset)
    #print(f'inference for {ngraphs} events is {itime} s (that\'s {itime/ngraphs} s/graph')

    #df = []
    #for ib, batch in enumerate(tqdm.tqdm(out)):
    #    for data in batch.to_data_list():
    #        #print(data)
    #        df.append(plot.to_dataframe(data))
    #        #df.append(plot.to_dataframe_evt(data))
    #        #print(df)
    #df = pd.concat(df)
    #df.to_hdf(args.outfile, 'hits', format='table')
    ##df.to_hdf(args.outfile, 'evt', format='table')

if __name__ == '__main__':
    args = configure()
    savejit(args)
