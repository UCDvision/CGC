import argparse
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.models.resnet as resnet
import datasets as pointing_datasets
from torchray.attribution.excitation_backprop import contrastive_excitation_backprop
from torchray.attribution.excitation_backprop import update_resnet
from models import resnet_to_fc
from torchray.attribution.common import get_pointing_gradient


""" 
    Here, we evaluate the content heatmap (Excitation Backprop heatmap within object bounding box) on imagenet dataset.
"""

model_names = ['resnet18', 'resnet50']

parser = argparse.ArgumentParser(description='Pointing game evaluation for ImageNet using Contrastive Excitation Backprop')
parser.add_argument('data', metavar='DIR', help='path to dataset')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet18)')
parser.add_argument('-j', '--workers', default=16, type=int, metavar='N',
                    help='number of data loading workers (default: 16)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N', help='mini-batch size (default: 96)')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('-g', '--num-gpus', default=1, type=int,
                    metavar='N', help='number of GPUs to match (default: 4)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--input_resize', default=224, type=int,
                    metavar='N', help='Resize for smallest side of input (default: 224)')


def main():
    global args
    args = parser.parse_args()

    if args.pretrained:
        print("=> using pre-trained model '{}'".format(args.arch))
        model = resnet.__dict__[args.arch](pretrained=True)
    else:
        print("=> creating model '{}'".format(args.arch))
        model = resnet.__dict__[args.arch]()
    model = torch.nn.DataParallel(model)

    if args.resume:
        print("=> loading checkpoint '{}'".format(args.resume))
        checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint['state_dict'])

    if (not args.resume) and (not args.pretrained):
        assert False, "Please specify either the pre-trained model or checkpoint for evaluation"

    model = model._modules['module']

    model = resnet_to_fc(model)
    model.avgpool = torch.nn.AvgPool2d((7, 7), stride=1)
    model = update_resnet(model, debug=True)
    model = model.cuda()
    cudnn.benchmark = False

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    # In the first version, we will not resize the images. We feed the full image and use AdaptivePooling before FC.
    # We will resize Gradcam heatmap to image size and compare the actual bbox co-ordinates
    val_dataset = pointing_datasets.ImageNetDetection(args.data,
                                       transform=transforms.Compose([
                                           # transforms.Resize((224, 224)),
                                           transforms.Resize(args.input_resize),
                                           transforms.ToTensor(),
                                           normalize,
                                       ]))

    # we set batch size=1 since we are loading full resolution images.
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=1, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    validate_multi(val_loader, val_dataset, model)


def validate_multi(val_loader, val_dataset, model):
    batch_time = AverageMeter()
    heatmap_inside_gt_mask = AverageMeter()

    # switch to evaluate mode
    model.eval()
    # We do not need grads for the parameters.
    for param in model.parameters():
        param.requires_grad_(False)

    # prepare vis layer, contrast layer and probes
    contrast_layer = 'avgpool'
    
    zero_saliency_count = 0
    total_count = 0
    end = time.time()
    for i, (images, annotation, targets) in enumerate(val_loader):
        total_count += 1
        images = images.cuda(non_blocking=True)
        targets = targets.cuda(non_blocking=True)
        # we assume batch size == 1 and unwrap the first elem of every list in annotation object
        annotation = unwrap_dict(annotation)
        image_size = val_dataset.as_image_size(annotation)

        class_id = targets[0].item()

        saliency = contrastive_excitation_backprop(model, images, class_id,
                                                   saliency_layer='layer3',
                                                   contrast_layer=contrast_layer,
                                                   resize=image_size,
                                                   get_backward_gradient=get_pointing_gradient
                                                   )
        saliency = saliency.squeeze()    # since we have batch size==1

        resized_saliency = saliency.data.cpu().numpy()

        if np.isnan(resized_saliency).any():
            zero_saliency_count += 1
            continue
        spatial_sum = resized_saliency.sum()
        if spatial_sum <= 0:
            zero_saliency_count += 1
            continue
        resized_saliency = resized_saliency / spatial_sum

        # Now, we obtain the mask corresponding to the ground truth bounding boxes
        # Skip if all boxes for class_id are marked difficult.
        objs = annotation['annotation']['object']
        if not isinstance(objs, list):
            objs = [objs]
        objs = [obj for obj in objs if pointing_datasets._IMAGENET_CLASS_TO_INDEX[obj['name']] == class_id]
        if all([bool(int(obj['difficult'])) for obj in objs]):
            continue
        gt_mask = pointing_datasets.imagenet_as_mask(annotation, class_id)
        gt_mask = gt_mask.type(torch.ByteTensor)
        gt_mask = gt_mask.cpu().data.numpy()
        gcam_inside_gt_mask = gt_mask * resized_saliency
        total_gcam_inside_gt_mask = gcam_inside_gt_mask.sum()
        heatmap_inside_gt_mask.update(total_gcam_inside_gt_mask)

        if i % 1000 == 0:
            print('\nCurr % of heatmap inside GT mask: {:.4f} ({:.4f})'.format(heatmap_inside_gt_mask.val * 100,
                                                                                 heatmap_inside_gt_mask.avg * 100))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

    print('\n\n% of heatmap inside GT mask: {:.4f}'.format(heatmap_inside_gt_mask.avg * 100))
    print('\n Zero Saliency found for {} / {} images.'.format(zero_saliency_count, total_count))

    return


def compute_gradcam(output, feats, target):
    """
    Compute the gradcam for the top predicted category
    :param output:
    :param feats:
    :return:
    """
    eps = 1e-8
    relu = nn.ReLU(inplace=True)

    target = target.cpu().numpy()
    # target = np.argmax(output.cpu().data.numpy(), axis=-1)
    one_hot = np.zeros((output.shape[0], output.shape[-1]), dtype=np.float32)
    indices_range = np.arange(output.shape[0])
    one_hot[indices_range, target[indices_range]] = 1
    one_hot = torch.from_numpy(one_hot)
    one_hot.requires_grad = True

    # Compute the Grad-CAM for the original image
    one_hot_cuda = torch.sum(one_hot.cuda() * output)
    dy_dz1, = torch.autograd.grad(one_hot_cuda, feats, grad_outputs=torch.ones(one_hot_cuda.size()).cuda(),
                                  retain_graph=True, create_graph=True)
    gcam512_1 = dy_dz1 * feats
    gradcam = gcam512_1.sum(dim=1)
    gradcam = relu(gradcam)
    spatial_sum1 = gradcam.sum(dim=[1, 2]).unsqueeze(-1).unsqueeze(-1)
    gradcam = (gradcam / (spatial_sum1 + eps)) + eps

    return gradcam


def unwrap_dict(dict_object):
    new_dict = {}
    for k, v in dict_object.items():
        if k == 'object':
            new_v_list = []
            for elem in v:
                new_v_list.append(unwrap_dict(elem))
            new_dict[k] = new_v_list
            continue
        if isinstance(v, dict):
            new_v = unwrap_dict(v)
        elif isinstance(v, list) and len(v) == 1:
            new_v = v[0]
            # if isinstance(new_v, dict):
            #     new_v = unwrap_dict(new_v)
        else:
            new_v = v
        new_dict[k] = new_v
    return new_dict


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


if __name__ == '__main__':
    main()
