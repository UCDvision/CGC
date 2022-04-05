import numpy as np
import torch
import torch.nn.functional as F


def compute_gradcam(output, feats , target, relu):
    """
    Compute the gradcam for the top predicted category
    :param output: the model output before softmax
    :param feats: the feature output from the desired layer to be used for computing Grad-CAM
    :param target: The target category to be used for computing Grad-CAM
    :return:
    """

    one_hot = np.zeros((output.shape[0], output.shape[-1]), dtype=np.float32)
    indices_range = np.arange(output.shape[0])
    one_hot[indices_range, target[indices_range]] = 1
    one_hot = torch.from_numpy(one_hot)
    one_hot.requires_grad = True

    # Compute the Grad-CAM for the original image
    one_hot_cuda = torch.sum(one_hot.cuda() * output)
    dy_dz1, = torch.autograd.grad(one_hot_cuda, feats, grad_outputs=torch.ones(one_hot_cuda.size()).cuda(),
                                    retain_graph=True, create_graph=True)
    # Changing to dot product of grad and features to preserve grad spatial locations
    gcam512_1 = dy_dz1 * feats
    gradcam = gcam512_1.sum(dim=1)
    gradcam = relu(gradcam)

    return gradcam


def compute_gradcam_mask(images_outputs, images_feats , target, relu):
    """
    This function computes the grad-cam, upsamples it to the image size and normalizes the Grad-CAM mask.
    """
    eps = 1e-8
    gradcam_mask = compute_gradcam(images_outputs, images_feats , target, relu)
    gradcam_mask = gradcam_mask.unsqueeze(1)
    gradcam_mask = F.interpolate(gradcam_mask, size=224, mode='bilinear')
    gradcam_mask = gradcam_mask.squeeze()
    # normalize the gradcam mask to sum to 1
    gradcam_mask_sum = gradcam_mask.sum(dim=[1, 2]).unsqueeze(-1).unsqueeze(-1)
    gradcam_mask = (gradcam_mask / (gradcam_mask_sum + eps)) + eps

    return gradcam_mask


def perform_gradcam_aug(orig_gradcam_mask, aug_params_dict):
    """
    This function uses the augmentation params per batch element and manually applies to the 
    grad-cam mask to obtain the corresponding augmented grad-cam mask.
    """
    transforms_i = aug_params_dict['transforms_i']
    transforms_j = aug_params_dict['transforms_j']
    transforms_h = aug_params_dict['transforms_h']
    transforms_w = aug_params_dict['transforms_w']
    hor_flip = aug_params_dict['hor_flip']
    gpu_batch_len = transforms_i.shape[0]
    augmented_orig_gradcam_mask = torch.zeros_like(orig_gradcam_mask).cuda()
    for b in range(gpu_batch_len):
        # convert orig_gradcam_mask to image
        orig_gcam = orig_gradcam_mask[b]
        orig_gcam = orig_gcam[transforms_i[b]: transforms_i[b] + transforms_h[b],
                    transforms_j[b]: transforms_j[b] + transforms_w[b]]
        # We use torch functional to resize without breaking the graph
        orig_gcam = orig_gcam.unsqueeze(0).unsqueeze(0)
        orig_gcam = F.interpolate(orig_gcam, size=224, mode='bilinear')
        orig_gcam = orig_gcam.squeeze()
        if hor_flip[b]:
            orig_gcam = orig_gcam.flip(-1)
        augmented_orig_gradcam_mask[b, :, :] = orig_gcam[:, :]
    return augmented_orig_gradcam_mask