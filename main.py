### importing the necessary modules. This project has been completed using the pytorch framework
import argparse
import os
import glob
import time
import numpy as np
import pandas as pd
from PIL import Image
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
from skimage.color import rgb2lab, lab2rgb

import torch
from torch import nn, optim
from torchvision import transforms
from torchvision.utils import make_grid
from torch.utils.data import Dataset, DataLoader
from torch.nn import DataParallel
from torch import hub

from fastai.vision.learner import create_body
from torchvision.models.resnet import resnet18
from fastai.vision.models.unet import DynamicUnet

from dataset import make_dataloaders
from models import *
from utils import *
from loss import *

import GPUtil

## Arguments
parser = argparse.ArgumentParser(description="Image Colourization with GANs.")
parser.add_argument('--net', default="resnet-18",
                    help="The Generator network architecture, it should be either of baseline, resnet-18, vgg-16, inception")#, convnext .") 
					## this allows us to create a generator network with different backbones.
parser.add_argument("--dataset", type=str, help="The root directory of the Coco dataset.")
parser.add_argument("--GAN_Mode", default="vanilla", type=str, help="to set the GAN criterion. If \"vanilla\", criteion is BCEwithLogits, if lsgan, criterion is \"MSE\", if PSNR, criterion is \"PSNR\" ")
#parser.add_argument("--weight_initializer", default="norm", type=str, help="Weight initialization method to use, it should be either of norm,xavier,kaiming.")
#parser.add_argument("--loss", type=str, default="PSNR,MSE", help="The loss used to train the nretwork. Should be PSNR,MSE or L1,L2.")
parser.add_argument("--cpt_dir", default="checkpoint_logs", type=str, help="The directory to store check points.")
parser.add_argument("--vis_dir", default="visualization_logs", type=str, help="The directory to store visualizations through out the job.")
parser.add_argument("--op_dir", default="output_logs", type=str, help="The directory to store outputs suchs as loss metrics through out the job.")
#parser.add_argument("--NUM_GPUS", default="4", type=int, help="NUM_GPUs.")
parser.add_argument("--BATCH_SIZE", default="16", type=int, help="BATCH_SIZE.")
parser.add_argument("--NUM_EPOCHS", default="16", type=int, help="NUM_EPOCHS.")
parser.add_argument("--LEARNING_RATE", default="16", type=float, help="LEARNING_RATE.")
#parser.add_argument("--Retrain", default="False", type=bool, help="retraining a previously trained model")
parser.add_argument("--pPrev", default="", type=str, help="path to a previously trained model")
parser.add_argument("--path_net_g", default="", type=str, help="path to a pretrained Net_G")
args = parser.parse_args()

#setting device to cuda if GPU is available, else to cpu.
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

## Train function - 
def train_model(model, train_dl, epochs, display_every=200):
    data = next(iter(val_dl)) # getting a batch for visualizing the model output after fixed intrvals
    log ={'loss_D_fake': [],
            'loss_D_real': [],
            'loss_D': [],
            'loss_G_GAN': [],
            'loss_G_L1': [],
            'loss_G': [],
            'time':[]
          }
    for e in range(epochs):
        start = time.time()
        loss_meter_dict = create_loss_meters() # function returing a dictionary of objects to log the losses of the complete network per each iteration
        for data in tqdm(train_dl):
            model.setup_input(data) 
            model.optimize()
            update_losses(model, loss_meter_dict, count=data['L'].size(0)) # function updating the log objects
            if e % 20 == 0:
                visualize(model, data, path = vis_logs+"/"+str(e)+"_vis.png")
                torch.save(model.state_dict(), cpt_logs+"/"+str(e)+'_gan.pt_')
        print(f"\nEpoch {e+1}/{epochs}")
        log = log_results(log,loss_meter_dict) # function to print and return the averaged losses across all iterations
        log['time'].append(start-time.time())
    ##storing the loss logs as csv file
    log_DF = pd.DataFrame(log)
    log_DF.to_csv(op_logs+"/"+args.net+"_output_logs.csv")

## function to pretrain the generator model 
def pretrain_generator(net_G, train_dl, opt, criterion, epochs):
    log = {}
    for e in range(epochs):
        loss_meter = AverageMeter()
        for data in tqdm(train_dl):
            L, ab = data['L'].to(device), data['ab'].to(device)
            preds = net_G(L)
            loss = criterion(preds, ab)
            opt.zero_grad()
            loss.backward()
            opt.step()
            
            loss_meter.update(loss.item(), L.size(0))
            
        print(f"Epoch {e + 1}/{epochs}")
        print(f"Loss: {loss_meter.avg:.5f}")
        log[str(epochs)]=loss_meter.avg
    
    log_DF = pd.DataFrame(log, index=[0])
    log_DF.to_csv(op_logs+"/"+args.net+"_Gen_logs.csv")
    
    path_net_g = cpt_logs+"/"+f"{args.net}"+"-unet.pt"
    torch.save(net_G.state_dict(), path_net_g)
    
    return path_net_g

## Main function

if __name__ == '__main__':
    path = args.dataset  ## path to the dataset.
    vis_logs = args.vis_dir    ## path to the visualization logs.
    cpt_logs = args.cpt_dir	   ## path to the checkpoint logs.
    op_logs = args.op_dir	   ## path to the output logs.

    hub.set_dir(cpt_logs)   ## setting the path to hub directory to checkpoint logs. Hub directory stores the pretrained models downloaded from torch.hub

    paths = glob.glob(path + "/*.jpg") # Grabbing all the image file names from the specified path
    np.random.seed(123)
    paths_subset = np.random.choice(paths, 10_000, replace=False) # choosing 10000 random images
    rand_idxs = np.random.permutation(10_000)
    train_idxs = rand_idxs[:8000] ## the first 8000 images are chosen as the training set
    val_idxs = rand_idxs[8000:] ## the last 2000 images are chosen as the validation set
    train_paths = paths_subset[train_idxs]
    val_paths = paths_subset[val_idxs]
    print("count of paths to images in training set: "+f'{len(train_paths)}'+"\ncount of paths to images in validation set: "+f'{len(val_paths)}')
    
    ### visualizing the original dataset
    _, axes = plt.subplots(4, 4, figsize=(10, 10))
    for ax, img_path in zip(axes.flatten(), train_paths):
        ax.imshow(Image.open(img_path))
        ax.axis("off")
    plt.savefig(vis_logs+"/Original_images.jpeg")
    
    ## Creating the dataloaders
    train_dl = make_dataloaders(args.BATCH_SIZE,paths=train_paths, split='train')
    val_dl = make_dataloaders(paths=val_paths, split='val')
    
    data = next(iter(train_dl))
    Ls, abs_ = data['L'], data['ab']
    print(Ls.shape, abs_.shape)
    print("count images in training set: "+f'{len(train_dl)}'+"\ncount of images in validation set"+f'{len(val_dl)}')
    
    ## Creating the GAN module.
    ## Visualizing the patch_discriminator and its output shape	
    discriminator = PatchDiscriminator(3)
    dummy_input = torch.randn(16, 3, 256, 256) # batch_size, channels, size, size
    out = discriminator(dummy_input)
    out.shape
    print("Discriminator")
    print(discriminator)
    print("output shape")
    print(out.shape)
    
    ### criterion 
    criterion = nn.L1Loss()  

    lr = args.LEARNING_RATE

    path_net_g = args.path_net_g
    
    print("creating generator network with : ",args.net)
    ## Creating the model
    if args.net == "baseline":
        net_G = None
        path_net_g = None
    elif args.net == "resnet-18":
        net_G = build_res_unet(n_input=1, n_output=2, size=256)
        if path_net_g == "":
            opt = optim.Adam(net_G.parameters(), lr=lr)
            path_net_g = pretrain_generator(net_G, train_dl, opt, criterion, 2)
    elif args.net == "vgg-16":
        net_G = build_vgg_unet(n_input=1, n_output=2, size=256)
        if path_net_g == "":
            opt = optim.Adam(net_G.parameters(), lr=lr)
            path_net_g = pretrain_generator(net_G, train_dl, opt, criterion, 20)
    elif args.net == "inception":	
        net_G = build_inception_unet(n_input=1, n_output=2, size=256)
        if path_net_g == "":
            opt = optim.Adam(net_G.parameters(), lr=lr)
            path_net_g = pretrain_generator(net_G, train_dl, opt, criterion, 20)
    #elif args.net == "convnext":
    #    net_G = build_ConvNext_unet(n_input=1, n_output=2, size=256)
    #    opt = optim.Adam(net_G.parameters(), lr=lr)
    #    path_net_g = pretrain_generator(net_G, train_dl, opt, criterion, 20)
    
    if net_G != None:
        net_G.load_state_dict(torch.load(path_net_g, map_location=device))
    #print(args.Retrain) 
    g_mode = args.GAN_Mode
    model = MainModel(net_G=net_G,gan_mode=g_mode)
    print("GAN Model created.")
    #print("Model parallelized over"+f'{torch.cuda.device_count()}'+'devices')
    #model = DataParallel(model)
    print("starting training")
    #with model.distrib_ctx():
    if args.pPrev != "" :
        model.load_state_dict(
                torch.load(args.pPrev,
                    map_location=device
                    )
                )
    train_model(model, train_dl, args.NUM_EPOCHS)
    print("saving model at "+cpt_logs+'/'+f'{args.net}'+'Final_gan.pt')
    torch.save(model.state_dict(), cpt_logs+'/'+f'{args.net}'+'Final_gan.pt')

    ### Show GPU Utilization
    print("checking GPU Utilization")
    GPUtil.showUtilization()

