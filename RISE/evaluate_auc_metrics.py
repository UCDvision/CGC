## Code to evaluate the pre-trained model and our CGC trained model with Insertion AUC score.

import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision.datasets as datasets
import torchvision.models as models
import torch.nn.functional as F
from utils import *
from evaluation import CausalMetric, auc, gkern
from pytorch_grad_cam import GradCAM
import argparse

parser = argparse.ArgumentParser(description='PyTorch AUC Metric Evaluation')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--ckpt-path', dest='ckpt_path', type=str, help='path to checkpoint file')

def main():
    args = parser.parse_args()

    cudnn.benchmark = True

    scores = {'del': [], 'ins': []}
    if args.pretrained:
        net = models.resnet50(pretrained=True)
    else:
        net = models.resnet50()
        state_dict = torch.load(args.ckpt_path)['state_dict']

        # remove the module prefix if model was saved with DataParallel
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        # load params
        net.load_state_dict(state_dict)

    target_layer = net.layer4[-1]
    cam = GradCAM(model=net, target_layer=target_layer, use_cuda=True)

    # we process the imagenet 50k val images in 10 set of 5k each and compute mean
    for i in range(10):
        auc_score = get_auc_per_data_subset(i, net, cam)
        scores['ins'].append(auc_score)
        print('Finished evaluating the insertion metrics...')

    print('----------------------------------------------------------------')
    print('Final:\nInsertion - {:.5f}'.format(np.mean(scores['ins'])))


def get_auc_per_data_subset(range_index, net, cam):
    batch_size = 100
    data_loader = torch.utils.data.DataLoader(
        dataset=datasets.ImageFolder('/nfs3/datasets/imagenet/val/', preprocess),
        batch_size=batch_size, shuffle=False,
        num_workers=8, pin_memory=True, sampler=RangeSampler(range(5000 * range_index, 5000 * (range_index + 1))))

    net = net.train()

    images = []
    targets = []
    gcam_exp = []

    for j, (img, trg) in enumerate(tqdm(data_loader, total=len(data_loader), desc='Loading images')):
        grayscale_gradcam = cam(input_tensor=img, target_category=trg)
        for k in range(batch_size):
            images.append(img[k])
            targets.append(trg[k])
            gcam_exp.append(grayscale_gradcam[k])

    images = torch.stack(images).cpu().numpy()
    gcam_exp = np.stack(gcam_exp)
    images = np.asarray(images)
    gcam_exp = np.asarray(gcam_exp)

    images = images.reshape((-1, 3, 224, 224))
    gcam_exp = gcam_exp.reshape((-1, 224, 224))
    print('Finished obtaining CAM')

    model = nn.Sequential(net, nn.Softmax(dim=1))
    model = model.eval()
    model = model.cuda()

    for p in model.parameters():
        p.requires_grad = False

    # To use multiple GPUs
    ddp_model = nn.DataParallel(model)

    # we use blur as the substrate function
    klen = 11
    ksig = 5
    kern = gkern(klen, ksig)
    # Function that blurs input image
    blur = lambda x: F.conv2d(x, kern, padding=klen // 2)

    insertion = CausalMetric(ddp_model, 'ins', 224 * 8, substrate_fn=blur)

    # Evaluate insertion
    h = insertion.evaluate(torch.from_numpy(images.astype('float32')), gcam_exp, batch_size)

    model = model.train()
    for p in model.parameters():
        p.requires_grad = True

    return auc(h.mean(1))


if __name__ == '__main__':
    main()
