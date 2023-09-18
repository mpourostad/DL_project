import argparse
import PIL
import torch
from matplotlib import pyplot as plt
from torchvision import transforms
import matplotlib.image

from models import *
from utils import *
from loss import *
import time
from torch import hub

parser = argparse.ArgumentParser(description="Image Colourization with GANs.")
parser.add_argument('--net', default="resnet-18", type=str, help="type of net_G") ## Path to the model.
parser.add_argument('--pathNetG', default="", type=str, help="path to the net_G model") ## Path to the NetG.
parser.add_argument('--pathGAN', default="", type=str, help="path to the final model") ## Path to the model.
parser.add_argument('--pathImg', default="", type=str, help="path to the test image") ## Path to the test image.
parser.add_argument('--pathOP', default="", type=str, help="path to the output directory") ## Path to the Output directory.
args = parser.parse_args()

if __name__ == '__main__':

    hub.set_dir("Results/checkpoints")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if args.net == "baseline":
        net_G = None
    elif args.net == "resnet-18":
        net_G = build_res_unet(n_input=1, n_output=2, size=256)
    elif args.net == "vgg-16":
        net_G = build_vgg_unet(n_input=1, n_output=2, size=256)
    elif args.net == "inception":	
        net_G = build_inception_unet(n_input=1, n_output=2, size=256)

    net_G.load_state_dict(
            torch.load(
                args.pathNetG,
                map_location=device
                )
            )
    model = MainModel(net_G)
    # You first need to download the final_model_weights.pt file from my drive
    # using the command: gdown --id 1lR6DcS4m5InSbZ5y59zkH2mHt_4RQ2KV
    model.load_state_dict(
        torch.load(
            args.pathGAN,
            map_location=device
        )
    )
    path = args.pathImg
    img = PIL.Image.open(path)
    img = img.resize((256, 256))
    # to make it between -1 and 1
    img = transforms.ToTensor()(img)[:1] * 2. - 1.
    model.eval()
    with torch.no_grad():
        preds = model.net_G(img.unsqueeze(0).to(device))
    colorized = lab_to_rgb(img.unsqueeze(0), preds.cpu())[0]

    plt.imshow(colorized)

    matplotlib.image.imsave(args.pathOP+f"/inference_{time.time()}.png", colorized)
    #plt.savefig(args.pathOP+f"inference_{time.time()}.png")
