import os, sys
import numpy as np
import imageio
import json
import random
import cv2
import math
import time

import torch
from torch.utils.tensorboard import SummaryWriter


from torch.autograd import Variable
import pytorch_ssim
import torch.nn as nn
import flax
import jax
from jax import numpy as jnp
import torch.nn.functional as F
from tqdm import tqdm, trange
import tensorflow as tf

import matplotlib.pyplot as plt

from run_nerf_helpers import *

from load_llff import load_llff_data
from load_deepvoxels import load_dv_data
from load_blender import load_blender_data
from load_LINEMOD import load_LINEMOD_data


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
np.random.seed(0)
DEBUG = False


def batchify(fn, chunk):
    """Constructs a version of 'fn' that applies to smaller batches.
    """
    if chunk is None:
        return fn
    def ret(inputs):
        return torch.cat([fn(inputs[i:i+chunk]) for i in range(0, inputs.shape[0], chunk)], 0)
    return ret

#netchunk 缩小了16倍
def run_network(inputs, viewdirs, fn, embed_fn, embeddirs_fn, netchunk=1024*32):
    """Prepares inputs and applies network 'fn'.
    """
    inputs_flat = torch.reshape(inputs, [-1, inputs.shape[-1]])
    embedded = embed_fn(inputs_flat)

    if viewdirs is not None:
        input_dirs = viewdirs[:,None].expand(inputs.shape)
        input_dirs_flat = torch.reshape(input_dirs, [-1, input_dirs.shape[-1]])
        embedded_dirs = embeddirs_fn(input_dirs_flat)
        embedded = torch.cat([embedded, embedded_dirs], -1)

    outputs_flat = batchify(fn, netchunk)(embedded)
    outputs = torch.reshape(outputs_flat, list(inputs.shape[:-1]) + [outputs_flat.shape[-1]])
    return outputs

# def compute_multiscale_ssim(image1: np.ndarray, image2: np.ndarray):
def compute_multiscale_ssim(image1, image2):
    """Compute the multiscale SSIM metric."""
    # ycord = np.vectorize(np.int)
    # image1=ycord(image1)
    # image2=ycord(image2)
    image1= image1.astype('float64')
    image2 = image2.astype('float64')
    image1 = tf.convert_to_tensor(image1)
    image2 = tf.convert_to_tensor(image2)

    return tf.image.ssim_multiscale(image1, image2, max_val=1.0)

def compute_psnr(mse):
  """Compute psnr value given mse (we assume the maximum pixel value is 1).

  Args:
    mse: float, mean square error of pixels.

  Returns:
    psnr: float, the psnr value.
  """
  return -10. * np.log(mse) / np.log(10.)
def ssim(img1,img2):
    for i in range(img1.shape[0]):
        img1[i] = torch.from_numpy(np.rollaxis(img1[i], 2)).float().unsqueeze(0)/255.0
        img1[i] = Variable(img1[i], requires_grad=False)  # torch.Size([256, 256, 3])
        img2[i] = torch.from_numpy(np.rollaxis(img2[i], 2)).float().unsqueeze(0)/255.0
        img2[i] = Variable( img2[i], requires_grad = False)
        ssim_value[i] = pytorch_ssim.ssim(img1[i], img2[i]).item()
    ssim_value2=np.mean(ssim_value)
    return ssim_value2

#compute lpips
__LPIPS__ = {}
def init_lpips(net_name, device):
  assert net_name in ['alex', 'vgg']
  import lpips
  print(f'init_lpips: lpips_{net_name}')
  return lpips.LPIPS(net=net_name, version='0.1').eval().to(device)


def rgb_lpips(np_gt, np_im, net_name, device):
  if net_name not in __LPIPS__:
    __LPIPS__[net_name] = init_lpips(net_name, device)
  gt = torch.from_numpy(np_gt).permute([2, 0, 1]).contiguous().to(device)
  im = torch.from_numpy(np_im).permute([2, 0, 1]).contiguous().to(device)
  gt = gt.float()
  im = im.float()
  return __LPIPS__[net_name](gt, im, normalize=True).item()

def batchify_rays(rays_flat, chunk=1024*8, **kwargs):
    """Render rays in smaller minibatches to avoid OOM.
    """
    all_ret = {}
    for i in range(0, rays_flat.shape[0], chunk):
        ret = render_rays(rays_flat[i:i+chunk], **kwargs)
        for k in ret:
            if k not in all_ret:
                all_ret[k] = []
            all_ret[k].append(ret[k])

    all_ret = {k : torch.cat(all_ret[k], 0) for k in all_ret}
    return all_ret

#chunk 缩小了16倍
def render(H, W, K, chunk=1024*64, rays=None, c2w=None, ndc=True,
                  near=0., far=1.,
                  use_viewdirs=False, c2w_staticcam=None,
                  **kwargs):
    """Render rays
    Args:
      H: int. Height of image in pixels.
      W: int. Width of image in pixels.
      focal: float. Focal length of pinhole camera.
      chunk: int. Maximum number of rays to process simultaneously. Used to
        control maximum memory usage. Does not affect final results.
      rays: array of shape [2, batch_size, 3]. Ray origin and direction for
        each example in batch.
      c2w: array of shape [3, 4]. Camera-to-world transformation matrix.
      ndc: bool. If True, represent ray origin, direction in NDC coordinates.
      near: float or array of shape [batch_size]. Nearest distance for a ray.
      far: float or array of shape [batch_size]. Farthest distance for a ray.
      use_viewdirs: bool. If True, use viewing direction of a point in space in model.
      c2w_staticcam: array of shape [3, 4]. If not None, use this transformation matrix for
       camera while using other c2w argument for viewing directions.
    Returns:
      rgb_map: [batch_size, 3]. Predicted RGB values for rays.
      disp_map: [batch_size]. Disparity map. Inverse of depth.
      acc_map: [batch_size]. Accumulated opacity (alpha) along a ray.
      extras: dict with everything returned by render_rays().
    """
    if c2w is not None:
        # special case to render full image
        rays_o, rays_d = get_rays(H, W, K, c2w)
    else:
        # use provided ray batch
        rays_o = rays[:,0]
        rays_d = rays[:,1]

    if use_viewdirs:
        # provide ray directions as input
        viewdirs = rays_d
        if c2w_staticcam is not None:
            # special case to visualize effect of viewdirs
            rays_o, rays_d = get_rays(H, W, K, c2w_staticcam)
        #print("viewdir.shape0", viewdirs.shape)
        viewdirs=torch.tensor(viewdirs,dtype=torch.float)
        #print("viewdir.shape1", viewdirs.shape)
        #print("norm", torch.norm(viewdirs, dim=-1, keepdim=True))
        viewdirs = viewdirs / torch.norm(viewdirs, dim=-1, keepdim=True)
        #print("viewdir.shape",viewdirs.shape)
        viewdirs = torch.reshape(viewdirs, [-1,3]).float()

    sh = rays_d.shape # [..., 3]
    if ndc:
        # for forward facing scenes
        rays_o, rays_d = ndc_rays(H, W, K[0][0], 1., rays_o, rays_d)

    # Create ray batch
    rays_o = torch.reshape(rays_o, [-1,3]).float()
    rays_d = torch.reshape(rays_d, [-1,3]).float()

    near, far = near * torch.ones_like(rays_d[...,:1]), far * torch.ones_like(rays_d[...,:1])
    rays = torch.cat([rays_o, rays_d, near, far], -1)
    if use_viewdirs:
        rays = torch.cat([rays, viewdirs], -1)

    # Render and reshape
    all_ret = batchify_rays(rays, chunk, **kwargs)
    for k in all_ret:
        k_sh = list(sh[:-1]) + list(all_ret[k].shape[1:])
        all_ret[k] = torch.reshape(all_ret[k], k_sh)

    k_extract = ['rgb_map', 'disp_map', 'acc_map']
    ret_list = [all_ret[k] for k in k_extract]
    ret_dict = {k : all_ret[k] for k in all_ret if k not in k_extract}
    return ret_list + [ret_dict]


def render_path(render_poses, hwf, K, chunk, render_kwargs, gt_imgs=None, savedir=None, render_factor=0):

    H, W, focal = hwf

    if render_factor!=0:
        # Render downsampled for speed
        H = H//render_factor
        W = W//render_factor
        focal = focal/render_factor

    rgbs = []
    disps = []

    t = time.time()
    for i, c2w in enumerate(tqdm(render_poses)):
        print(i, time.time() - t)
        t = time.time()
        rgb, disp, acc, _ = render(H, W, K, chunk=chunk, c2w=c2w[:3,:4], **render_kwargs)

        rgbs.append(rgb.cpu().numpy())
        disps.append(disp.cpu().numpy())
        if i==0:
            print(rgb.shape, disp.shape)

        """
        if gt_imgs is not None and render_factor==0:
            p = -10. * np.log10(np.mean(np.square(rgb.cpu().numpy() - gt_imgs[i])))
            print(p)
        """

        if savedir is not None:
            rgb8 = to8b(rgbs[-1])
            filename = os.path.join(savedir, '{:03d}.png'.format(i))
            #filename = os.path.join("/Users/jiawei/yufei/nerf-pytorch/logs/blender_paper_medical/render_test_result", 'testright{:03d}.png'.format(i))
            imageio.imwrite(filename, rgb8)


    rgbs = np.stack(rgbs, 0)
    disps = np.stack(disps, 0)

    return rgbs, disps
def get_rbg(render_poses, hwf, K, chunk, render_kwargs, gt_imgs=None, savedir=None, render_factor=0):


    H, W, focal = hwf
    lpips_alexs = np.array([])
    lpips_vggs = np.array([])
    mses = np.array([])
    psnrs = np.array([])
    ssims = np.array([])
    if render_factor!=0:
        # Render downsampled for speed
        H = H//render_factor
        W = W//render_factor
        focal = focal/render_factor

    rgbs = []
    mses= []
    disps = []

    t = time.time()
    for i, c2w in enumerate(tqdm(render_poses)):
        print(i, time.time() - t)
        t = time.time()
        print(chunk)
        rgb, disp, acc, _ = render(H, W, K, chunk=chunk, c2w=c2w[:3,:4], **render_kwargs)


        rgbs.append(rgb.cpu().numpy())
        rgb8 = to8b(rgbs[-1])
        # filename = os.path.join("/Users/jiawei/yufei/nerf-pytorch/logs/blender_paper_medical/render_test_result",
        #                         'testright{:03d}.png'.format(i))
        main_path=os.path.join(savedir,"render")
        if not os.path.exists(main_path):  # 如果路径不存在
            os.makedirs(main_path)
        main_path = os.path.join(savedir, "gt")
        if not os.path.exists(main_path):  # 如果路径不存在
            os.makedirs(main_path)
        filename = os.path.join(savedir,"render",
                                'testright{:03d}.png'.format(i))

        imageio.imwrite(filename, rgb8)
        rgb9=imageio.v2.imread(filename)

        # filename1 = os.path.join("/Users/jiawei/yufei/nerf-pytorch/logs/blender_paper_medical/gt",
        #                         'gt{:03d}.png'.format(i))
        filename1 = os.path.join(savedir,"gt",
                                 'gt{:03d}.png'.format(i))
        imageio.imwrite(filename1,gt_imgs[i])
        gt = imageio.v2.imread(filename1)

        rgb9=rgb9/255.0
        gt=gt/255.0

        rgb9 = torch.Tensor(rgb9)
        gt = torch.Tensor(gt)

        mse = ((rgb9 - gt) ** 2).mean()
        psnr = mse2psnr(mse)
        print('mse', mse)
        print('psnr', psnr)

        lpips_alex = rgb_lpips(np.array(rgb9), np.array(gt), net_name='alex', device="cpu")
        lpips_vgg = rgb_lpips(np.array(rgb9), np.array(gt), net_name='vgg', device="cpu")
        lpips_alexs = np.append(lpips_alexs, lpips_alex)
        lpips_vggs=np.append(lpips_vggs,lpips_vgg)
        print('lpips_alex', lpips_alex)
        print('lpips_vgg', lpips_vgg)


        ssim = compute_multiscale_ssim(np.array(rgb9), np.array(gt))
        print('ssim',ssim)
        mses = np.append(mses, mse)
        psnrs = np.append(psnrs, psnr)
        ssims = np.append(ssims, ssim)

        if i==0:
            print(rgb.shape, disp.shape)

        """
        if gt_imgs is not None and render_factor==0:
            p = -10. * np.log10(np.mean(np.square(rgb.cpu().numpy() - gt_imgs[i])))
            print(p)
        """
    lpips_alexs = np.mean(lpips_alexs)
    lpips_vggs = np.mean(lpips_vggs)
    ssims      = np.mean(ssims)
    psnrs =np.mean(psnrs)
    mses =np.mean(mses)
    return lpips_alexs,lpips_vggs,ssims,psnrs,mses

def create_nerf(args):
    """Instantiate NeRF's MLP model.
    """
    embed_fn, input_ch = get_embedder(args.multires, args.i_embed) #3D位置编码最大频率的log，默认10  ，默认0，为默认的位置编码
                                                                   #返回编码好的数组以及对应的维度，
    input_ch_views = 0
    embeddirs_fn = None
    if args.use_viewdirs:
        embeddirs_fn, input_ch_views = get_embedder(args.multires_views, args.i_embed)
    output_ch = 5 if args.N_importance > 0 else 4
    skips = [4]
    model = NeRF(D=args.netdepth, W=args.netwidth,
                 input_ch=input_ch, output_ch=output_ch, skips=skips,
                 input_ch_views=input_ch_views, use_viewdirs=args.use_viewdirs).to(device)
    grad_vars = list(model.parameters())

    model_fine = None
    if args.N_importance > 0: #N_importance为128
        model_fine = NeRF(D=args.netdepth_fine, W=args.netwidth_fine,  #depth=8  width=128
                          input_ch=input_ch, output_ch=output_ch, skips=skips,
                          input_ch_views=input_ch_views, use_viewdirs=args.use_viewdirs).to(device)
        grad_vars += list(model_fine.parameters())

    network_query_fn = lambda inputs, viewdirs, network_fn : run_network(inputs, viewdirs, network_fn,
                                                                embed_fn=embed_fn,
                                                                embeddirs_fn=embeddirs_fn,
                                                                netchunk=args.netchunk)

    # Create optimizer
    optimizer = torch.optim.Adam(params=grad_vars, lr=args.lrate, betas=(0.9, 0.999))

    start = 0
    basedir = args.basedir
    expname = args.expname

    ##########################

    # Load checkpoints
    if args.ft_path is not None and args.ft_path!='None':
        ckpts = [args.ft_path]
    else:
        ckpts = [os.path.join(basedir, expname, f) for f in sorted(os.listdir(os.path.join(basedir, expname))) if 'tar' in f]

    print('Found ckpts', ckpts)
    if len(ckpts) > 0 and not args.no_reload:
        ckpt_path = ckpts[-1]
        print('Reloading from', ckpt_path)
        ckpt = torch.load(ckpt_path)

        start = ckpt['global_step']
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])

        # Load model
        model.load_state_dict(ckpt['network_fn_state_dict'])
        if model_fine is not None:
            model_fine.load_state_dict(ckpt['network_fine_state_dict'])

    ##########################

    render_kwargs_train = {
        'network_query_fn' : network_query_fn,
        'perturb' : args.perturb,
        'N_importance' : args.N_importance,
        'network_fine' : model_fine,
        'N_samples' : args.N_samples,
        'network_fn' : model,
        'use_viewdirs' : args.use_viewdirs,
        'white_bkgd' : args.white_bkgd,
        'raw_noise_std' : args.raw_noise_std,
    }

    # NDC only good for LLFF-style forward facing data
    if args.dataset_type != 'llff' or args.no_ndc:
        print('Not ndc!')
        render_kwargs_train['ndc'] = False
        render_kwargs_train['lindisp'] = args.lindisp

    render_kwargs_test = {k : render_kwargs_train[k] for k in render_kwargs_train}
    render_kwargs_test['perturb'] = False
    render_kwargs_test['raw_noise_std'] = 0.

    return render_kwargs_train, render_kwargs_test, start, grad_vars, optimizer
    #start 为开始的状态，0，或者是之后保存好的节点，frad_var 为参数


def raw2outputs(raw, z_vals, rays_d, raw_noise_std=0, white_bkgd=False, pytest=False):
    """Transforms model's predictions to semantically meaningful values.
    Args:
        raw: [num_rays, num_samples along ray, 4]. Prediction from model.
        z_vals: [num_rays, num_samples along ray]. Integration time.
        rays_d: [num_rays, 3]. Direction of each ray.
    Returns:
        rgb_map: [num_rays, 3]. Estimated RGB color of a ray.
        disp_map: [num_rays]. Disparity map. Inverse of depth map.
        acc_map: [num_rays]. Sum of weights along each ray.
        weights: [num_rays, num_samples]. Weights assigned to each sampled color.
        depth_map: [num_rays]. Estimated distance to object.
    """
    raw2alpha = lambda raw, dists, act_fn=F.relu: 1.-torch.exp(-act_fn(raw)*dists)

    dists = z_vals[...,1:] - z_vals[...,:-1] #两个点之间的间隔
    dists = torch.cat([dists, torch.Tensor([1e10]).expand(dists[...,:1].shape)], -1)  # [N_rays, N_samples]

    dists = dists * torch.norm(rays_d[...,None,:], dim=-1)

    rgb = torch.sigmoid(raw[...,:3])  # [N_rays, N_samples, 3]
    noise = 0.
    if raw_noise_std > 0.:
        noise = torch.randn(raw[...,3].shape) * raw_noise_std

        # Overwrite randomly sampled data if pytest
        if pytest:
            np.random.seed(0)
            noise = np.random.rand(*list(raw[...,3].shape)) * raw_noise_std
            noise = torch.Tensor(noise)

    alpha = raw2alpha(raw[...,3] + noise, dists)  # [N_rays, N_samples]
    # weights = alpha * tf.math.cumprod(1.-alpha + 1e-10, -1, exclusive=True)
    weights = alpha * torch.cumprod(torch.cat([torch.ones((alpha.shape[0], 1)), 1.-alpha + 1e-10], -1), -1)[:, :-1]
    rgb_map = torch.sum(weights[...,None] * rgb, -2)  # [N_rays, 3]

    depth_map = torch.sum(weights * z_vals, -1)
    disp_map = 1./torch.max(1e-10 * torch.ones_like(depth_map), depth_map / torch.sum(weights, -1))
    acc_map = torch.sum(weights, -1)

    if white_bkgd:
        rgb_map = rgb_map + (1.-acc_map[...,None])

    return rgb_map, disp_map, acc_map, weights, depth_map


def render_rays(ray_batch,
                network_fn,
                network_query_fn,
                N_samples,
                retraw=False,
                lindisp=False,
                perturb=0.,
                N_importance=0,
                network_fine=None,
                white_bkgd=False,
                raw_noise_std=0.,
                verbose=False,
                pytest=False):
    """Volumetric rendering.
    Args:
      ray_batch: array of shape [batch_size, ...]. All information necessary
        for sampling along a ray, including: ray origin, ray direction, min
        dist, max dist, and unit-magnitude viewing direction.
      network_fn: function. Model for predicting RGB and density at each point
        in space.
      network_query_fn: function used for passing queries to network_fn.
      N_samples: int. Number of different times to sample along each ray.
      retraw: bool. If True, include model's raw, unprocessed predictions.
      lindisp: bool. If True, sample linearly in inverse depth rather than in depth.
      perturb: float, 0 or 1. If non-zero, each ray is sampled at stratified
        random points in time.
      N_importance: int. Number of additional times to sample along each ray.
        These samples are only passed to network_fine.
      network_fine: "fine" network with same spec as network_fn.
      white_bkgd: bool. If True, assume a white background.
      raw_noise_std: ...
      verbose: bool. If True, print more debugging info.
    Returns:
      rgb_map: [num_rays, 3]. Estimated RGB color of a ray. Comes from fine model.
      disp_map: [num_rays]. Disparity map. 1 / depth.
      acc_map: [num_rays]. Accumulated opacity along each ray. Comes from fine model.
      raw: [num_rays, num_samples, 4]. Raw predictions from model.
      rgb0: See rgb_map. Output for coarse model.
      disp0: See disp_map. Output for coarse model.
      acc0: See acc_map. Output for coarse model.
      z_std: [num_rays]. Standard deviation of distances along ray for each
        sample.
    """
    N_rays = ray_batch.shape[0]
    rays_o, rays_d = ray_batch[:,0:3], ray_batch[:,3:6] # [N_rays, 3] each
    viewdirs = ray_batch[:,-3:] if ray_batch.shape[-1] > 8 else None
    bounds = torch.reshape(ray_batch[...,6:8], [-1,1,2])
    near, far = bounds[...,0], bounds[...,1] # [-1,1]

    t_vals = torch.linspace(0., 1., steps=N_samples)
    if not lindisp:
        z_vals = near * (1.-t_vals) + far * (t_vals)
    else:
        z_vals = 1./(1./near * (1.-t_vals) + 1./far * (t_vals))

    z_vals = z_vals.expand([N_rays, N_samples])

    if perturb > 0.:
        # get intervals between samples
        mids = .5 * (z_vals[...,1:] + z_vals[...,:-1])
        upper = torch.cat([mids, z_vals[...,-1:]], -1)
        lower = torch.cat([z_vals[...,:1], mids], -1)
        # stratified samples in those intervals
        t_rand = torch.rand(z_vals.shape)

        # Pytest, overwrite u with numpy's fixed random numbers
        if pytest:
            np.random.seed(0)
            t_rand = np.random.rand(*list(z_vals.shape))
            t_rand = torch.Tensor(t_rand)

        z_vals = lower + (upper - lower) * t_rand

    pts = rays_o[...,None,:] + rays_d[...,None,:] * z_vals[...,:,None] # [N_rays, N_samples, 3]


#     raw = run_network(pts)
    raw = network_query_fn(pts, viewdirs, network_fn)
    rgb_map, disp_map, acc_map, weights, depth_map = raw2outputs(raw, z_vals, rays_d, raw_noise_std, white_bkgd, pytest=pytest)

    if N_importance > 0:

        rgb_map_0, disp_map_0, acc_map_0 = rgb_map, disp_map, acc_map

        z_vals_mid = .5 * (z_vals[...,1:] + z_vals[...,:-1])
        z_samples = sample_pdf(z_vals_mid, weights[...,1:-1], N_importance, det=(perturb==0.), pytest=pytest)
        z_samples = z_samples.detach()

        z_vals, _ = torch.sort(torch.cat([z_vals, z_samples], -1), -1)
        pts = rays_o[...,None,:] + rays_d[...,None,:] * z_vals[...,:,None] # [N_rays, N_samples + N_importance, 3]

        run_fn = network_fn if network_fine is None else network_fine
#         raw = run_network(pts, fn=run_fn)
        raw = network_query_fn(pts, viewdirs, run_fn)

        rgb_map, disp_map, acc_map, weights, depth_map = raw2outputs(raw, z_vals, rays_d, raw_noise_std, white_bkgd, pytest=pytest)

    ret = {'rgb_map' : rgb_map, 'disp_map' : disp_map, 'acc_map' : acc_map}
    if retraw:
        ret['raw'] = raw
    if N_importance > 0:
        ret['rgb0'] = rgb_map_0
        ret['disp0'] = disp_map_0
        ret['acc0'] = acc_map_0
        ret['z_std'] = torch.std(z_samples, dim=-1, unbiased=False)  # [N_rays]

    for k in ret:
        if (torch.isnan(ret[k]).any() or torch.isinf(ret[k]).any()) and DEBUG:
            print(f"! [Numerical Error] {k} contains nan or inf.")

    return ret


def config_parser():

    import configargparse
    parser = configargparse.ArgumentParser()
    parser.add_argument('--config', is_config_file=True, 
                        help='config file path')
    parser.add_argument("--expname", type=str, 
                        help='experiment name')
    parser.add_argument("--basedir", type=str, default='./logs/', 
                        help='where to store ckpts and logs')
    parser.add_argument("--datadir", type=str, default='./data/llff/fern', 
                        help='input data directory')


    # training options
    parser.add_argument("--netdepth", type=int, default=8, 
                        help='layers in network')
    parser.add_argument("--netwidth", type=int, default=256, 
                        help='channels per layer')
    parser.add_argument("--netdepth_fine", type=int, default=8, 
                        help='layers in fine network')
    parser.add_argument("--netwidth_fine", type=int, default=256, 
                        help='channels per layer in fine network')
    parser.add_argument("--N_rand", type=int, default=32*32*4,  #1024*4 缩小了4倍
                        help='batch size (number of random rays per gradient step)')
    parser.add_argument("--lrate", type=float, default=5e-4, 
                        help='learning rate')
    parser.add_argument("--lrate_decay", type=int, default=250, 
                        help='exponential learning rate decay (in 1000 steps)')
    parser.add_argument("--chunk", type=int, default=1024*32,
                        help='number of rays processed in parallel, decrease if running out of memory,default=1024*32')
    parser.add_argument("--netchunk", type=int, default=1024*64,
                        help='number of pts sent through network in parallel, decrease if running out of memory,default=1024*64')
    parser.add_argument("--no_batching", action='store_true', 
                        help='only take random rays from 1 image at a time')
    parser.add_argument("--no_reload", action='store_true', 
                        help='do not reload weights from saved ckpt')
    parser.add_argument("--ft_path", type=str, default=None, 
                        help='specific weights npy file to reload for coarse network')

    # rendering options
    parser.add_argument("--N_samples", type=int, default=64, 
                        help='number of coarse samples per ray')
    parser.add_argument("--N_importance", type=int, default=0,
                        help='number of additional fine samples per ray')
    parser.add_argument("--perturb", type=float, default=1.,
                        help='set to 0. for no jitter, 1. for jitter')
    parser.add_argument("--use_viewdirs", action='store_true', 
                        help='use full 5D input instead of 3D')
    parser.add_argument("--i_embed", type=int, default=0, 
                        help='set 0 for default positional encoding, -1 for none')
    parser.add_argument("--multires", type=int, default=10, 
                        help='log2 of max freq for positional encoding (3D location)')
    parser.add_argument("--multires_views", type=int, default=4, 
                        help='log2 of max freq for positional encoding (2D direction)')
    parser.add_argument("--raw_noise_std", type=float, default=0., 
                        help='std dev of noise added to regularize sigma_a output, 1e0 recommended')

    parser.add_argument("--render_only", action='store_true', 
                        help='do not optimize, reload weights and render out render_poses path')
    parser.add_argument("--get_eval", action='store_true',
                        help='do not optimize, caculate the LPIPS,PSRN,SSIM evaluation')
    parser.add_argument("--render_test", action='store_true', 
                        help='render the test set instead of render_poses path')
    parser.add_argument("--render_factor", type=int, default=0, 
                        help='downsampling factor to speed up rendering, set 4 or 8 for fast preview')

    # training options
    parser.add_argument("--precrop_iters", type=int, default=0,
                        help='number of steps to train on central crops')
    parser.add_argument("--precrop_frac", type=float,
                        default=.5, help='fraction of img taken for central crops') 

    # dataset options
    parser.add_argument("--dataset_type", type=str, default='llff', 
                        help='options: llff / blender / deepvoxels')
    parser.add_argument("--testskip", type=int, default=1,
                        help='will load 1/N images from test/val sets, useful for large datasets like deepvoxels')

    ## deepvoxels flags
    parser.add_argument("--shape", type=str, default='greek', 
                        help='options : armchair / cube / greek / vase')

    ## blender flags
    parser.add_argument("--white_bkgd", action='store_true', 
                        help='set to render synthetic data on a white bkgd (always use for dvoxels)')
    parser.add_argument("--half_res", action='store_true', 
                        help='load blender synthetic data at 400x400 instead of 800x800')

    ## llff flags
    parser.add_argument("--factor", type=int, default=8, 
                        help='downsample factor for LLFF images')
    parser.add_argument("--no_ndc", action='store_true', 
                        help='do not use normalized device coordinates (set for non-forward facing scenes)')
    parser.add_argument("--lindisp", action='store_true', 
                        help='sampling linearly in disparity rather than depth')
    parser.add_argument("--spherify", action='store_true', 
                        help='set for spherical 360 scenes')
    parser.add_argument("--llffhold", type=int, default=1,
                        help='will take every 1/N images as LLFF test set, paper uses 8')

    # logging/saving options
    parser.add_argument("--i_print",   type=int, default=100, 
                        help='frequency of console printout and metric loggin')
    parser.add_argument("--i_img",     type=int, default=500, 
                        help='frequency of tensorboard image logging')
    parser.add_argument("--i_weights", type=int, default=10000, 
                        help='frequency of weight ckpt saving')
    parser.add_argument("--i_testset", type=int, default=200000,
                        help='frequency of testset saving')
    parser.add_argument("--i_video",   type=int, default=200000,
                        help='frequency of render_poses video saving')
    parser.add_argument("--i_render_test", type=int, default=200000,
                        help='caculate the parameter and render test')
    parser.add_argument("--i_render_train", type=int, default=200000,
                        help='caculate the parameter and render test')
    return parser


def train():

        parser = config_parser()
        args = parser.parse_args()
        summary_dir = '/Users/jiawei/yufei/nerf-pytorch/summaries/eval'
        writer = SummaryWriter(str(summary_dir))

        # Load data
        K = None
        if args.dataset_type == 'llff':
            images, poses, bds, render_poses, i_test = load_llff_data(args.datadir, args.factor,
                                                                      recenter=True, bd_factor=.75,
                                                                      spherify=args.spherify)
            hwf = poses[0,:3,-1]
            poses = poses[:,:3,:4]
            print('Loaded llff', images.shape, render_poses.shape, hwf, args.datadir)
            if not isinstance(i_test, list):
                i_test = [i_test]

            if args.llffhold > 0:
                print('Auto LLFF holdout,', args.llffhold)
                i_test = np.arange(images.shape[0])[::args.llffhold]

            i_val = i_test
            i_train = np.array([i for i in np.arange(int(images.shape[0])) if
                            (i not in i_test and i not in i_val)])

            print('DEFINING BOUNDS')
            if args.no_ndc:
                near = np.ndarray.min(bds) * .9
                far = np.ndarray.max(bds) * 1.

            else:
                near = 0.
                far = 1.
            print('NEAR FAR', near, far)

        elif args.dataset_type == 'blender':
            images, poses, render_poses, hwf, i_split = load_blender_data(args.datadir, args.half_res, args.testskip)
            print('Loaded blender', images.shape, render_poses.shape, hwf, args.datadir)
            #(1201, 475, 475, 4) 1201张图片，475*475大小，4维度，最后一维没东西
            i_train, i_val, i_test = i_split

            near = 0.
            far = 6.

            if args.white_bkgd:
                images = images[...,:3]*images[...,-1:] + (1.-images[...,-1:])
            else:
                images = images[...,:3]  #最后一维好像没东西
            #images=images[0:120]
            # for i in range(120):
            #     save_path='/Users/jiawei/yufei/nerf-pytorch/data/nerf_synthetic/medical2/train'
            #     cv2.imwrite(save_path, images[i])
            #poses=poses[0:120]
            #render_poses=render_poses[0:120]
            #hwf = hwf[0:120]
            # i_train=i_train[0:110]
            # i_test=i_test[0:10]
            # 
            # train_image=images[i_train]
            # test_image = images[i_test]
            # val_image=images[i_val]
            # for i in range(110):
            #
            #     root = '/Users/jiawei/yufei/nerf-pytorch/data/nerf_synthetic/medical2/train'
            #     save_path = root +'/'+ str(i) + '.png'
            #     if not os.path.exists(root):  # 如果路径不存在
            #         os.makedirs(root)
            #     imgs = to8b(train_image[i])
            #     cv2.imwrite(save_path,imgs)
            # for i in range(10):
            #     root = '/Users/jiawei/yufei/nerf-pytorch/data/nerf_synthetic/medical2/test'
            #     save_path = root + '/'+str(i) + '.png'
            #     if not os.path.exists(root):  # 如果路径不存在
            #         os.makedirs(root)
            #     imgs = to8b(test_image[i])
            #     cv2.imwrite(save_path, imgs)
            #
            # root = '/Users/jiawei/yufei/nerf-pytorch/data/nerf_synthetic/medical2/val'
            # save_path = root + '/'+str('0') + '.png'
            # if not os.path.exists(root):  # 如果路径不存在
            #     os.makedirs(root)
            # imgs= to8b(val_image)
            # cv2.imwrite(save_path, imgs)
            # print('images_shape',images.shape)
            # print('poses',poses.shape)
            # print('render_poses',render_poses.shape)
            #print('hwf', hwf.shape)
            # print('i_train', i_train.shape)
            # print('i_test', i_test.shape)
            # print('i_train_first_end',i_train[0])
            # print('i_train_first_end',i_train[-1])
            # print('i_test_first_end', i_test[0])
            # print('i_test_first_end', i_test[-1])


        elif args.dataset_type == 'LINEMOD':
            images, poses, render_poses, hwf, K, i_split, near, far = load_LINEMOD_data(args.datadir, args.half_res, args.testskip)
            print(f'Loaded LINEMOD, images shape: {images.shape}, hwf: {hwf}, K: {K}')
            print(f'[CHECK HERE] near: {near}, far: {far}.')
            i_train, i_val, i_test = i_split

            if args.white_bkgd:
                images = images[...,:3]*images[...,-1:] + (1.-images[...,-1:])
            else:
                images = images[...,:3]

        elif args.dataset_type == 'deepvoxels':

            images, poses, render_poses, hwf, i_split = load_dv_data(scene=args.shape,
                                                                     basedir=args.datadir,
                                                                     testskip=args.testskip)

            print('Loaded deepvoxels', images.shape, render_poses.shape, hwf, args.datadir)
            i_train, i_val, i_test = i_split

            hemi_R = np.mean(np.linalg.norm(poses[:,:3,-1], axis=-1))
            near = hemi_R-1.
            far = hemi_R+1.

        else:
            print('Unknown dataset type', args.dataset_type, 'exiting')
            return

        # Cast intrinsics to right types
        H, W, focal = hwf
        H, W = int(H), int(W)
        hwf = [H, W, focal]
        # 计算内参矩阵K
        if K is None:
            K = np.array([
                [focal, 0, 0.5*W],
                [0, focal, 0.5*H],
                [0, 0, 1]
            ])

        if args.render_test:
            render_poses = np.array(poses[i_test])
            render_poses_train=np.array(poses[i_train])
        if args.get_eval:
            render_poses = np.array(poses[i_test])
            render_poses_train=np.array(poses[i_train])
        # Create log dir and copy the config file
        basedir = args.basedir
        expname = args.expname
        os.makedirs(os.path.join(basedir, expname), exist_ok=True)
        f = os.path.join(basedir, expname, 'args.txt')
        with open(f, 'w') as file:
            for arg in sorted(vars(args)):
                attr = getattr(args, arg)
                file.write('{} = {}\n'.format(arg, attr))
        if args.config is not None:
            f = os.path.join(basedir, expname, 'config.txt')
            with open(f, 'w') as file:
                file.write(open(args.config, 'r').read())

        # Create nerf model
        render_kwargs_train, render_kwargs_test, start, grad_vars, optimizer = create_nerf(args)
        global_step = start

        bds_dict = {
            'near' : near,
            'far' : far,
        }
        render_kwargs_train.update(bds_dict)
        render_kwargs_test.update(bds_dict)

        # Move testing data to GPU
        render_poses = torch.Tensor(render_poses).to(device)
        render_poses_train = np.array(poses[i_train])
        render_poses_train = torch.Tensor(render_poses_train).to(device)
        # Short circuit if only rendering out from trained model
        if args.render_only:
            print('RENDER ONLY')
            with torch.no_grad():
                if args.render_test:
                    # render_test switches to test poses
                    images_test = images[i_test]
                    images_train = images[i_train]
                else:
                    # Default is smoother render_poses path
                    images = None

                print(i_test.shape)
                testsavedir = os.path.join(basedir, expname, 'renderonly_{}_{:06d}'.format('test' if args.render_test else 'path', start))
                trainsavedir = os.path.join(basedir, expname,'renderonly_{}_{:06d}'.format('train' , start))
                os.makedirs(testsavedir, exist_ok=True)
                os.makedirs(trainsavedir, exist_ok=True)
                print('test poses shape', render_poses.shape)
                print('train poses shape', render_poses_train.shape)
                rgbs, _ = render_path(render_poses, hwf, K, args.chunk, render_kwargs_test, gt_imgs=images_test, savedir=testsavedir, render_factor=args.render_factor)
                rgbs2, _ = render_path(render_poses_train, hwf, K, args.chunk, render_kwargs_train, gt_imgs=images_train,
                                      savedir=trainsavedir, render_factor=args.render_factor)
                print('Done rendering', testsavedir)
                imageio.mimwrite(os.path.join(testsavedir, 'video.mp4'), to8b(rgbs), fps=30, quality=8)
                imageio.mimwrite(os.path.join(trainsavedir, 'video_train.mp4'), to8b(rgbs2), fps=30, quality=8)
                return
        if args.get_eval:
            print('get eval ONLY')
            with torch.no_grad():
                if args.get_eval:
                    # render_test switches to test poses
                    images_test = images[i_test]
                    images_train = images[i_train]



                print('test poses shape', render_poses.shape)
                print('train poses shape', render_poses_train.shape)

                trainsavedir = "./logs/blender_paper_medical/train2"
                if not os.path.exists(trainsavedir):  # 如果路径不存在
                    os.makedirs(trainsavedir)
                testsavedir= "./logs/blender_paper_medical/test2"
                test_lpip_alex,test_lpip_vgg,test_ssim,test_psnr,test_mse = get_rbg(render_poses, hwf, K, args.chunk, render_kwargs_test, gt_imgs=images_test,
                                                                                    savedir=testsavedir,
                                       render_factor=args.render_factor)

                train_lpip_alex,train_lpip_vgg ,train_ssim,train_psnr,train_mse= get_rbg(render_poses_train, hwf, K, args.chunk, render_kwargs_train,
                                       gt_imgs=images_train,
                                       savedir=trainsavedir, render_factor=args.render_factor)
                print('test_lpip_alex: ',test_lpip_alex)
                print('test_lpip_vgg: ', test_lpip_vgg)
                print('test_ssim: ', test_ssim)
                print('test_psnr: ', test_psnr)
                print('test_mse:' , test_mse)
                print('train_lpip_alex', train_lpip_alex)
                print('train_lpip_vgg', train_lpip_vgg)
                print('train_ssim: ', train_ssim)
                print('train_psnr: ', train_psnr)
                print('train_mse:', train_mse)
                return

        # Prepare raybatch tensor if batching random rays
        N_rand = args.N_rand
        use_batching = not args.no_batching
        if use_batching:
            # For random ray batching
            print('get rays')
            rays = np.stack([get_rays_np(H, W, K, p) for p in poses[:,:3,:4]], 0) # [N, ro+rd, H, W, 3] #返回射线的原点和方向
            print('done, concats')
            rays_rgb = np.concatenate([rays, images[:,None]], 1) # [N, ro+rd+rgb, H, W, 3]
            rays_rgb = np.transpose(rays_rgb, [0,2,3,1,4]) # [N, H, W, ro+rd+rgb, 3]
            rays_rgb = np.stack([rays_rgb[i] for i in i_train], 0) # train images only
            rays_rgb = np.reshape(rays_rgb, [-1,3,3]) # [(N-1)*H*W, ro+rd+rgb, 3]
            rays_rgb = rays_rgb.astype(np.float32)
            print('shuffle rays')
            np.random.shuffle(rays_rgb)

            print('done')
            i_batch = 0

        # Move training data to GPU
        if use_batching:
            images = torch.Tensor(images).to(device)
        poses = torch.Tensor(poses).to(device)
        if use_batching:
            rays_rgb = torch.Tensor(rays_rgb).to(device)


        # N_iters = 200000 + 1
        N_iters = 200000 + 1
        print('Begin')
        print('TRAIN views are', i_train)
        print('TEST views are', i_test)
        print('VAL views are', i_val)

        # Summary writers
        # writer = SummaryWriter(os.path.join(basedir, 'summaries', expname))

        start = start + 1
        for i in trange(start, N_iters):
            time0 = time.time()

            # Sample random ray batch
            if use_batching:
                # Random over all images
                batch = rays_rgb[i_batch:i_batch+N_rand] # [B, 2+1, 3*?]  [(N-1)*H*W, ro+rd+rgb, 3] 475*475*1000=220000 000
                #batch = (batch, 0, 1)
                #print(batch.shape)
                batch_rays, target_s = batch[:,:2], batch[:,2]
                #print(batch_rays.shape)
                i_batch += N_rand
                if i_batch >= rays_rgb.shape[0]:
                    print("Shuffle data after an epoch!")
                    rand_idx = torch.randperm(rays_rgb.shape[0]) #打乱这个数组的数据，将射线的顺序弄乱
                    rays_rgb = rays_rgb[rand_idx]
                    i_batch = 0
            else:
                # Random from one image
                img_i = np.random.choice(i_train)  #随机拿了一个图
                target = images[img_i]             #target为原图
                target = torch.Tensor(target).to(device)
                pose = poses[img_i, :3,:4]         #取出旋转平移矩阵，省略最后一行0001的

                if N_rand is not None:
                    rays_o, rays_d = get_rays(H, W, K, torch.Tensor(pose))  # (H, W, 3), (H, W, 3) 得到所有射线的起点以及方向在世界坐标系下

                    if i < args.precrop_iters:
                        dH = int(H//2 * args.precrop_frac) #相当于H/4 了
                        dW = int(W//2 * args.precrop_frac)
                        coords = torch.stack(                                    #相当于生成了一堆坐标 一个小正方形从左上角到右下角的坐标
                            torch.meshgrid(                                      #生成坐标网络
                                torch.linspace(H//2 - dH, H//2 + dH - 1, 2*dH),  #H/4,3/4H 挨个加1
                                torch.linspace(W//2 - dW, W//2 + dW - 1, 2*dW)
                            ), -1)
                        if i == start:
                            print(f"[Config] Center cropping of size {2*dH} x {2*dW} is enabled until iter {args.precrop_iters}")
                    else:
                        coords = torch.stack(torch.meshgrid(torch.linspace(0, H-1, H), torch.linspace(0, W-1, W)), -1)  # (H, W, 2)

                    coords = torch.reshape(coords, [-1,2])  # (H * W, 2)
                    select_inds = np.random.choice(coords.shape[0], size=[N_rand], replace=False)  # (N_rand,) #从坐标中随机选择1000个点
                    select_coords = coords[select_inds].long()  # (N_rand, 2)
                    rays_o = rays_o[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)
                    rays_d = rays_d[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)
                    batch_rays = torch.stack([rays_o, rays_d], 0)
                    target_s = target[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)
            print("到render")
            #####  Core optimization loop  #####
            rgb, disp, acc, extras = render(H, W, K, chunk=args.chunk, rays=batch_rays,
                                                    verbose=i < 10, retraw=True,
                                                    **render_kwargs_train)

            optimizer.zero_grad()
            img_loss = img2mse(rgb, target_s)

            # rgb1=rgb.unsqueeze(2)
            # rgb1=rgb1.cpu().detach().numpy()

            # target_s1=target_s.unsqueeze(2)
            # target_s1=target_s1.cpu().detach().numpy()
            # print(type(rgb))
            # print(rgb.shape)
            #
            # train_ssim= compute_multiscale_ssim(rgb1,target_s1)
            trans = extras['raw'][...,-1]
            loss = img_loss
            psnr = mse2psnr(img_loss)

            if 'rgb0' in extras:
                img_loss0 = img2mse(extras['rgb0'], target_s)
                loss = loss + img_loss0
                psnr0 = mse2psnr(img_loss0)

            loss.backward()
            optimizer.step()

            # NOTE: IMPORTANT!
            ###   update learning rate   ###
            decay_rate = 0.1
            decay_steps = args.lrate_decay * 1000
            new_lrate = args.lrate * (decay_rate ** (global_step / decay_steps))
            for param_group in optimizer.param_groups:
                param_group['lr'] = new_lrate
            ################################

            dt = time.time()-time0
            # print(f"Step: {global_step}, Loss: {loss}, Time: {dt}")
            #####           end            #####

            # Rest is logging
            if i % args.i_video == 0 and i > 0:
                # Turn on testing mode
                with torch.no_grad():
                    render_poses_train = np.array(poses[i_train])
                    render_poses_train  = torch.Tensor(render_poses_train).to(device)
                    rgbs1, disps1 = render_path(render_poses_train, hwf, K, args.chunk, render_kwargs_train)

                print('Done, saving', rgbs1.shape, disps1.shape)
                moviebase = os.path.join(basedir, expname, '{}_spiral_{:06d}_'.format(expname, i))
                imageio.mimwrite(moviebase + 'train_rgb.mp4', to8b(rgbs1), fps=15, quality=8)
                imageio.mimwrite(moviebase + 'train_disp1.mp4', to8b(disps1 / np.max(disps1)), fps=15, quality=8)
                break
            if i%args.i_weights==0:
                path = os.path.join(basedir, expname, '{:06d}.tar'.format(i))
                torch.save({
                    'global_step': global_step,
                    'network_fn_state_dict': render_kwargs_train['network_fn'].state_dict(),
                    'network_fine_state_dict': render_kwargs_train['network_fine'].state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }, path)
                print('Saved checkpoints at', path)
            # if i % args.i_render_train == 0 and i > 0:
            #     # Turn on testing mode
            #     with torch.no_grad():
            #         trainsavedir = "./logs/blender_paper_medical/train"
            #         if not os.path.exists(trainsavedir):  # 如果路径不存在
            #             os.makedirs(trainsavedir)
            #         images_train = images[i_train]
            #         # render_poses_train = np.array(poses[i_train])
            #         # render_poses_train = torch.Tensor(render_poses_train).to(device)
            #
            #
            #         train_lpip_alex, train_lpip_vgg, train_ssim, train_psnr, train_mse = get_rbg(render_poses_train,
            #                                                                                      hwf, K, args.chunk,
            #                                                                                      render_kwargs_train,
            #                                                                                      gt_imgs=images_train,
            #                                                                                      savedir=trainsavedir,
            #                                                                                      render_factor=args.render_factor)
            #         print('train_lpip_alex', train_lpip_alex)
            #         print('train_lpip_vgg', train_lpip_vgg)
            #         print('train_ssim: ', train_ssim)
            #         print('train_psnr: ', train_psnr)
            #         print('train_mse:', train_mse)
            #         writer.add_scalar(tag="lpips_alex/train", scalar_value=train_lpip_alex, global_step=i)
            #         writer.add_scalar(tag="lpips_vgg/train", scalar_value=train_lpip_vgg, global_step=i)
            #         writer.add_scalar(tag="ssim/train", scalar_value=train_ssim, global_step=i)
            #         writer.add_scalar(tag="psnr/train", scalar_value=train_psnr, global_step=i)
            #         writer.add_scalar(tag="mse/train", scalar_value=train_mse, global_step=i)
            #     # if args.use_viewdirs:
            #     #     render_kwargs_test['c2w_staticcam'] = render_poses[0][:3,:4]
            #     #     with torch.no_grad():
            #     #         rgbs_still, _ = render_path(render_poses, hwf, args.chunk, render_kwargs_test)
            #     #     render_kwargs_test['c2w_staticcam'] = None
            #     #     imageio.mimwrite(moviebase + 'rgb_still.mp4', to8b(rgbs_still), fps=30, quality=8)

            if i%args.i_video==0 and i > 0:
                # Turn on testing mode
                with torch.no_grad():
                    render_poses_test = np.array(poses[i_test])
                    render_poses_test = torch.Tensor(render_poses_test).to(device)
                    rgbs, disps = render_path(render_poses_test, hwf, K, args.chunk, render_kwargs_test)
                print('Done, saving', rgbs.shape, disps.shape)
                moviebase = os.path.join(basedir, expname, '{}_spiral_{:06d}_'.format(expname, i))
                imageio.mimwrite(moviebase + 'rgb.mp4', to8b(rgbs), fps=2, quality=8)
                imageio.mimwrite(moviebase + 'disp.mp4', to8b(disps / np.max(disps)), fps=2, quality=8)
                #10w次渲染一次存到了test中
            if i % args.i_render_test == 0 and i > 0:
                    # Turn on testing mode
                    with torch.no_grad():

                        images_test = images[i_test]
                        render_poses_test = np.array(poses[i_test])
                        render_poses_test = torch.Tensor(render_poses_test).to(device)
                        testsavedir = "./logs/blender_paper_medical/test"
                          # 文件保存路径，如果不存在就会被重建
                        if not os.path.exists(testsavedir):  # 如果路径不存在
                            os.makedirs(testsavedir)
                        test_lpip_alex, test_lpip_vgg, test_ssim, test_psnr, test_mse = get_rbg(render_poses_test, hwf, K,
                                                                                                args.chunk,
                                                                                                render_kwargs_test,
                                                                                                gt_imgs=images_test,
                                                                                                savedir=testsavedir,
                                                                                                render_factor=args.render_factor)

                        print('test_lpip_alex: ', test_lpip_alex)
                        print('test_lpip_vgg: ', test_lpip_vgg)
                        print('test_ssim: ', test_ssim)
                        print('test_psnr: ', test_psnr)
                        print('test_mse:', test_mse)
                        writer.add_scalar(tag="lpips_alex/test", scalar_value=test_lpip_alex, global_step=i)
                        writer.add_scalar(tag="lpips_vgg/test" , scalar_value=test_lpip_vgg, global_step=i)
                        writer.add_scalar(tag="ssim/test", scalar_value=test_ssim, global_step=i)
                        writer.add_scalar(tag="psnr/test", scalar_value=test_psnr, global_step=i)
                        writer.add_scalar(tag="mse/test" , scalar_value=test_mse , global_step=i)  #10

            if i % args.i_render_train == 0 and i > 0:
                # Turn on testing mode
                with torch.no_grad():
                    trainsavedir = "./logs/blender_paper_medical/train"
                    if not os.path.exists(trainsavedir):  # 如果路径不存在
                        os.makedirs(trainsavedir)
                    images_train = images[i_train]
                    render_poses_train = np.array(poses[i_train])
                    render_poses_train = torch.Tensor(render_poses_train).to(device)
                    train_lpip_alex, train_lpip_vgg, train_ssim, train_psnr, train_mse = get_rbg(render_poses_train,
                                                                                                 hwf, K, args.chunk,
                                                                                                 render_kwargs_train,
                                                                                                 gt_imgs=images_train,
                                                                                                 savedir=trainsavedir,
                                                                                                 render_factor=args.render_factor)
                    print('train_lpip_alex', train_lpip_alex)
                    print('train_lpip_vgg', train_lpip_vgg)
                    print('train_ssim: ', train_ssim)
                    print('train_psnr: ', train_psnr)
                    print('train_mse:', train_mse)
                    writer.add_scalar(tag="lpips_alex/train", scalar_value=train_lpip_alex, global_step=i)
                    writer.add_scalar(tag="lpips_vgg/train", scalar_value=train_lpip_vgg, global_step=i)
                    writer.add_scalar(tag="ssim/train", scalar_value=train_ssim, global_step=i)
                    writer.add_scalar(tag="psnr/train", scalar_value=train_psnr, global_step=i)
                    writer.add_scalar(tag="mse/train", scalar_value=train_mse, global_step=i)
                # if args.use_viewdirs:
                #     render_kwargs_test['c2w_staticcam'] = render_poses[0][:3,:4]
                #     with torch.no_grad():
                #         rgbs_still, _ = render_path(render_poses, hwf, args.chunk, render_kwargs_test)
                #     render_kwargs_test['c2w_staticcam'] = None
                #     imageio.mimwrite(moviebase + 'rgb_still.mp4', to8b(rgbs_still), fps=30, quality=8)
              #5w一次
            if i%args.i_testset==0 and i > 0:
                testsavedir = os.path.join(basedir, expname, 'testset_{:06d}'.format(i))
                os.makedirs(testsavedir, exist_ok=True)
                print('test poses shape', poses[i_test].shape)
                with torch.no_grad():
                    render_path(torch.Tensor(poses[i_test]).to(device), hwf, K, args.chunk, render_kwargs_test, gt_imgs=images[i_test], savedir=testsavedir)
                print('Saved test set')

            if i%args.i_print==0:

                tqdm.write(f"[TRAIN] Iter: {i} Loss: {loss.item()}  PSNR: {psnr.item()}  ")
                writer.add_scalar(tag="loss/train", scalar_value=loss.item(),global_step=i)
                writer.add_scalar(tag="PSNR/train", scalar_value=psnr.item(), global_step=i)
                #summary_writer.image(f'disparity-median/{tag}/{item_id}', disp_med_viz, step)
            """
                print(expname, i, psnr.numpy(), loss.numpy(), global_step.numpy())
                print('iter time {:.05f}'.format(dt))
    
                with tf.contrib.summary.record_summaries_every_n_global_steps(args.i_print):
                    tf.contrib.summary.scalar('loss', loss)
                    tf.contrib.summary.scalar('psnr', psnr)
                    tf.contrib.summary.histogram('tran', trans)
                    if args.N_importance > 0:
                        tf.contrib.summary.scalar('psnr0', psnr0)
    
    
                if i%args.i_img==0:
    
                    # Log a rendered validation view to Tensorboard
                    img_i=np.random.choice(i_val)
                    target = images[img_i]
                    pose = poses[img_i, :3,:4]
                    with torch.no_grad():
                        rgb, disp, acc, extras = render(H, W, focal, chunk=args.chunk, c2w=pose,
                                                            **render_kwargs_test)
    
                    psnr = mse2psnr(img2mse(rgb, target))
    
                    with tf.contrib.summary.record_summaries_every_n_global_steps(args.i_img):
    
                        tf.contrib.summary.image('rgb', to8b(rgb)[tf.newaxis])
                        tf.contrib.summary.image('disp', disp[tf.newaxis,...,tf.newaxis])
                        tf.contrib.summary.image('acc', acc[tf.newaxis,...,tf.newaxis])
    
                        tf.contrib.summary.scalar('psnr_holdout', psnr)
                        tf.contrib.summary.image('rgb_holdout', target[tf.newaxis])
    
    
                    if args.N_importance > 0:
    
                        with tf.contrib.summary.record_summaries_every_n_global_steps(args.i_img):
                            tf.contrib.summary.image('rgb0', to8b(extras['rgb0'])[tf.newaxis])
                            tf.contrib.summary.image('disp0', extras['disp0'][tf.newaxis,...,tf.newaxis])
                            tf.contrib.summary.image('z_std', extras['z_std'][tf.newaxis,...,tf.newaxis])
            """

            global_step += 1


if __name__=='__main__':
    since = time.time()
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

    train()
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
