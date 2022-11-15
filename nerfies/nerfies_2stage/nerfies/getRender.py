
import json
from typing import List, Tuple
import os
from absl import logging
import cv2
import numpy as np
import torch
import imageio
trans_t = lambda t : torch.Tensor([
    [1,0,0,0],
    [0,1,0,0],
    [0,0,1,t],
    [0,0,0,1]]).float()

rot_phi = lambda phi : torch.Tensor([
    [1,0,0,0],
    [0,np.cos(phi),-np.sin(phi),0],
    [0,np.sin(phi), np.cos(phi),0],
    [0,0,0,1]]).float()

rot_theta = lambda th : torch.Tensor([
    [np.cos(th),0,-np.sin(th),0],
    [0,1,0,0],
    [np.sin(th),0, np.cos(th),0],
    [0,0,0,1]]).float()


def pose_spherical(theta, phi, radius):
    c2w = trans_t(radius)
    c2w = rot_phi(phi/180.*np.pi) @ c2w
    c2w = rot_theta(theta/180.*np.pi) @ c2w
    c2w = torch.Tensor(np.array([[-1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]])) @ c2w
    return c2w


def getRender():
    N_perpose = 50
    N_allpose = 0
    angles_all = np.linspace(-180, 180, 10 + 1)
    # angles_all = np.linspace(-252, 108, 10 + 1)  # glove kuka
    # angles_all = np.linspace(-108, 252, 10 + 1) #daisy robot
    render_poses = []
    render_times = []

    # position 0
    angles = np.ones([N_perpose]) * angles_all[0]
    # angles = np.linspace(angles_all[0], angles_all[1], N_perpose)
    render_poses.append(torch.stack([pose_spherical(angle, -30.0, 4.0) for angle in angles]))
    render_times.append(torch.linspace(0., 1., N_perpose))
    N_allpose += len(angles)
    # position 1
    # angles = np.ones([30]) * angles_all[1]
    angles = np.linspace(angles_all[0], angles_all[1], 30)
    render_poses.append(torch.stack([pose_spherical(angle, -30.0, 4.0) for angle in angles]))
    render_times.append(torch.linspace(1., 0.5, len(angles)))
    N_allpose += len(angles)
    # position 360
    angles = np.linspace(-144, 144, 144 + 1)
    # angles = np.linspace(-72, 216, 144 + 1)  # robot daisy
    # angles = np.linspace(-216, 72, 144 + 1) #glove kuka
    # thetas1 = np.linspace(-30, -90, int(len(angles) / 2)+1)
    # thetas2 = np.linspace(-90, -30, int(len(angles) / 2)+1)
    # thetas = np.concatenate((thetas1, thetas2))
    # render_poses.append(torch.stack([pose_spherical(angle, theta, 3.0) for (angle, theta) in zip(angles, thetas)]))
    render_poses.append(torch.stack([pose_spherical(angle, -30.0, 4.0) for angle in angles]))
    render_times.append(torch.ones([len(angles)]) * 0.5)
    N_allpose += len(angles)
    # position -2
    # angles = np.ones([30]) * angles_all[-2]
    angles = np.linspace(angles_all[-2], angles_all[-1], 30)
    render_poses.append(torch.stack([pose_spherical(angle, -30.0, 4.0) for angle in angles]))
    render_times.append(torch.linspace(0.5, 0, len(angles)))
    N_allpose += len(angles)
    # # position -1
    # angles = np.ones([N_perpose]) * angles_all[-1]
    # render_poses.append(torch.stack([pose_spherical(angle, -30.0, 4.0) for angle in angles]))
    # render_times.append(torch.linspace(0., 1., N_perpose))
    # N_allpose += len(angles)
    # import pdb
    # pdb.set_trace()
    # for i in range(10):
    #     angles = np.ones([N_perpose]) * angles_all[i]
    #     render_poses.append(torch.stack([pose_spherical(angle, -30.0, 4.0) for angle in angles]))
    #     render_times.append(torch.linspace(0., 1., N_perpose))
    # if i % 2 == 0:
    #     render_times.append(torch.linspace(0., 1., N_perpose))
    # else:
    #     render_times.append(torch.linspace(1., 0., N_perpose))
    render_poses = torch.cat(render_poses).reshape([N_allpose, 4, 4])
    render_times = torch.cat(render_times).reshape([N_allpose])

    # Fix video_version2
    N_perpose = 50
    N_allpose = 0
    angles_all = np.linspace(-180, 180, 10 + 1)
    # angles_all = np.linspace(-252, 108, 10 + 1)  # glove
    # angles_all = np.linspace(-108, 252, 10 + 1)  # daisy robot
    # import pdb
    # pdb.set_trace()
    render_poses_fixview = []
    render_times_fixview = []

    # position 0
    angles = np.ones([N_perpose]) * angles_all[0]
    render_poses_fixview.append(torch.stack([pose_spherical(angle, -30.0, 4.0) for angle in angles]))
    render_times_fixview.append(torch.linspace(0., 1., N_perpose))
    N_allpose += len(angles)
    # angles = np.ones([30]) * angles_all[1]
    angles = np.linspace(angles_all[0], angles_all[1], 30)
    render_poses_fixview.append(torch.stack([pose_spherical(angle, -30.0, 4.0) for angle in angles]))
    render_times_fixview.append(torch.linspace(1., 0.5, len(angles)))
    N_allpose += len(angles)
    # position 360
    angles = np.linspace(-144, 144, 144 + 1)
    # angles = np.linspace(-72, 216, 144 + 1)  # robot daisy
    # angles = np.linspace(-216, 72, 144 + 1)  # glove kuka
    thetas1 = np.linspace(-30, -90, int(len(angles) / 2) + 1)
    thetas2 = np.linspace(-90, -30, int(len(angles) / 2) + 1)
    thetas = np.concatenate((thetas1, thetas2))
    render_poses_fixview.append(
        torch.stack([pose_spherical(angle, theta, 4.0) for (angle, theta) in zip(angles, thetas)]))
    # render_poses_fixview.append(torch.stack([pose_spherical(angle, -30.0, 4.0) for angle in angles]))
    render_times_fixview.append(torch.ones([len(angles)]) * 0.5)
    N_allpose += len(angles)
    # position -2
    # angles = np.ones([30]) * angles_all[-2]
    angles = np.linspace(angles_all[-2], angles_all[-1], 30)
    render_poses_fixview.append(torch.stack([pose_spherical(angle, -30.0, 4.0) for angle in angles]))
    render_times_fixview.append(torch.linspace(0.5, 0, len(angles)))
    N_allpose += len(angles)
    # # position -1
    # angles = np.ones([N_perpose]) * angles_all[-1]
    # render_poses_fixview.append(torch.stack([pose_spherical(angle, -30.0, 4.0) for angle in angles]))
    # render_times_fixview.append(torch.linspace(0., 1., N_perpose))
    # N_allpose += len(angles)

    render_poses_fixview = torch.cat(render_poses_fixview).reshape([N_allpose, 4, 4])
    render_times_fixview = torch.cat(render_times_fixview).reshape([N_allpose])

    #     angles = np.ones([40]) * (-180)
    #     # angles = np.array(
    #     # [-180., -180.,  -180., -180., -180., -180., -180., -180., -180.,
    #     # -180.,  -180.,  -180., -180., -180., -180., -180., -180., -180.,
    #     # -180.,  -180.,  -180., -180., -180., -180., -180., -180., -180.,
    #     # -180.,  -180.,  -180., -180., -180., -180., -180., -180., -180.,
    #     # -180.,  -180.,  -180., -180.])
    #     render_poses_fixview = torch.stack([pose_spherical(angle, -30.0, 4.0) for angle in angles], 0)
    # render_times_fixview = torch.linspace(0., 1., render_poses_fixview.shape[0])

    # Fix time and change view
    # import pdb
    # pdb.set_trace()
    angles = np.array(
        [-180., -171., -162., -153., -144., -135., -126., -117., -108.,
         -99., -90., -81., -72., -63., -54., -45., -36., -27.,
         -18., -9., 0., 9., 18., 27., 36., 45., 54.,
         63., 72., 81., 90., 99., 108., 117., 126., 135.,
         144., 153., 162., 171.])
    # phis = np.array(
    # [-180., -180.,  -180., -180., -180., -180., -180., -180., -180.,
    # -180.,  -180.,  -180., -180., -180., -180., -180., -180., -180.,
    # -180.,  -180.,  -180., -180., -180., -180., -180., -180., -180.,
    # -180.,  -180.,  -180., -180., -180., -180., -180., -180., -180.,
    # -180.,  -180.,  -180., -180.])
    render_poses_fixtime = torch.stack([pose_spherical(angle, -30.0, 4.0) for angle in angles], 0)
    render_times_fixtime = np.array(
        [0.0000, 0.0256, 0.0513, 0.0769, 0.1026, 0.1026, 0.1026, 0.1026, 0.2051,
         0.2051, 0.2051, 0.2051, 0.3077, 0.3077, 0.3077, 0.3077, 0.4103, 0.4103,
         0.4103, 0.4103, 0.5128, 0.5128, 0.5128, 0.5128, 0.6154, 0.6154, 0.6154,
         0.6154, 0.7179, 0.7179, 0.7179, 0.7179, 0.8205, 0.8205, 0.8205, 0.8205,
         0.9231, 0.9231, 0.9231, 0.9231])

    return render_poses, render_times, render_poses_fixview, render_times_fixview, render_poses_fixtime, render_times_fixtime

#
#
#
# def getRender():
#     N_perpose = 50
#     N_allpose = 0
#     angles_all = np.linspace(-252, 108, 10 + 1)
#     render_poses = []
#     render_times = []
#
#     # position 0
#     angles = np.ones([N_perpose]) * angles_all[0]
#     # angles = np.linspace(angles_all[0], angles_all[1], N_perpose)
#     render_poses.append(torch.stack([pose_spherical(angle, -30.0, 4.0) for angle in angles]))
#     render_times.append(torch.linspace(0., 1., N_perpose))
#     N_allpose += len(angles)
#     # position 1
#     # angles = np.ones([30]) * angles_all[1]
#     angles = np.linspace(angles_all[0], angles_all[1], 30)
#     render_poses.append(torch.stack([pose_spherical(angle, -30.0, 4.0) for angle in angles]))
#     render_times.append(torch.linspace(1., 0.5, len(angles)))
#     N_allpose += len(angles)
#     # position 360
#     angles = np.linspace(-216, 36, 126 + 1)
#     # thetas1 = np.linspace(-30, -90, int(len(angles) / 2)+1)
#     # thetas2 = np.linspace(-90, -30, int(len(angles) / 2)+1)
#     # thetas = np.concatenate((thetas1, thetas2))
#     # render_poses.append(torch.stack([pose_spherical(angle, theta, 3.0) for (angle, theta) in zip(angles, thetas)]))
#     render_poses.append(torch.stack([pose_spherical(angle, -30.0, 4.0) for angle in angles]))
#     render_times.append(torch.ones([len(angles)]) * 0.5)
#     N_allpose += len(angles)
#     # position -2
#     # angles = np.ones([30]) * angles_all[-2]
#     angles = np.linspace(angles_all[-3], angles_all[-2], 30)
#     render_poses.append(torch.stack([pose_spherical(angle, -30.0, 4.0) for angle in angles]))
#     render_times.append(torch.linspace(0.5, 0, len(angles)))
#     N_allpose += len(angles)
#     # position -1
#     angles = np.linspace(angles_all[-2], angles_all[-1], 50)
#     render_poses.append(torch.stack([pose_spherical(angle, -30.0, 4.0) for angle in angles]))
#     render_times.append(torch.linspace(0., 1., N_perpose))
#     N_allpose += len(angles)
#     # import pdb
#     # pdb.set_trace()
#     # for i in range(10):
#     #     angles = np.ones([N_perpose]) * angles_all[i]
#     #     render_poses.append(torch.stack([pose_spherical(angle, -30.0, 4.0) for angle in angles]))
#     #     render_times.append(torch.linspace(0., 1., N_perpose))
#     # if i % 2 == 0:
#     #     render_times.append(torch.linspace(0., 1., N_perpose))
#     # else:
#     #     render_times.append(torch.linspace(1., 0., N_perpose))
#     render_poses = torch.cat(render_poses).reshape([N_allpose, 4, 4])
#     render_times = torch.cat(render_times).reshape([N_allpose])
#     # render_poses = torch.stack([pose_spherical(angle, -30.0, 4.0) for angle in np.linspace(-180,180,400+1)[:-1]], 0)
#     # render_times = torch.linspace(0., 1., render_poses.shape[0])
#
#     N_perpose = 50
#     N_allpose = 0
#     angles_all = np.linspace(-252, 108, 10 + 1)
#     # import pdb
#     # pdb.set_trace()
#     render_poses_fixview = []
#     render_times_fixview = []
#
#     # position 0
#     angles = np.ones([N_perpose]) * angles_all[0]
#     render_poses_fixview.append(torch.stack([pose_spherical(angle, -30.0, 4.0) for angle in angles]))
#     render_times_fixview.append(torch.linspace(0., 1., N_perpose))
#     N_allpose += len(angles)
#     # angles = np.ones([30]) * angles_all[1]
#     angles = np.linspace(angles_all[0], angles_all[1], 30)
#     render_poses_fixview.append(torch.stack([pose_spherical(angle, -30.0, 4.0) for angle in angles]))
#     render_times_fixview.append(torch.linspace(1., 0.5, len(angles)))
#     N_allpose += len(angles)
#     # position 360
#     angles = np.linspace(-216, 36, 126 + 1)
#     thetas1 = np.linspace(-30, -90, int(len(angles) / 2) + 1)
#     thetas2 = np.linspace(-90, -30, int(len(angles) / 2) + 1)
#     thetas = np.concatenate((thetas1, thetas2))
#     render_poses_fixview.append(
#         torch.stack([pose_spherical(angle, theta, 4.0) for (angle, theta) in zip(angles, thetas)]))
#     # render_poses_fixview.append(torch.stack([pose_spherical(angle, -30.0, 4.0) for angle in angles]))
#     render_times_fixview.append(torch.ones([len(angles)]) * 0.5)
#     N_allpose += len(angles)
#     # position -2
#     # angles = np.ones([30]) * angles_all[-2]
#     angles = np.linspace(angles_all[-3], angles_all[-2], 30)
#     render_poses_fixview.append(torch.stack([pose_spherical(angle, -30.0, 4.0) for angle in angles]))
#     render_times_fixview.append(torch.linspace(0.5, 0, len(angles)))
#     N_allpose += len(angles)
#     # position -1
#     angles = np.linspace(angles_all[-2], angles_all[-1], 50)
#     render_poses_fixview.append(torch.stack([pose_spherical(angle, -30.0, 4.0) for angle in angles]))
#     render_times_fixview.append(torch.linspace(0., 1., N_perpose))
#     N_allpose += len(angles)
#
#     render_poses_fixview = torch.cat(render_poses_fixview).reshape([N_allpose, 4, 4])
#     render_times_fixview = torch.cat(render_times_fixview).reshape([N_allpose])
#
#     # Fix time and change vie
#     # import pdb
#     # pdb.set_trace()
#     angles = np.array(
#         [-180., -171., -162., -153., -144., -135., -126., -117., -108.,
#          -99., -90., -81., -72., -63., -54., -45., -36., -27.,
#          -18., -9., 0., 9., 18., 27., 36., 45., 54.,
#          63., 72., 81., 90., 99., 108., 117., 126., 135.,
#          144., 153., 162., 171.])
#     # phis = np.array(
#     # [-180., -180.,  -180., -180., -180., -180., -180., -180., -180.,
#     # -180.,  -180.,  -180., -180., -180., -180., -180., -180., -180.,
#     # -180.,  -180.,  -180., -180., -180., -180., -180., -180., -180.,
#     # -180.,  -180.,  -180., -180., -180., -180., -180., -180., -180.,
#     # -180.,  -180.,  -180., -180.])
#     render_poses_fixtime = torch.stack([pose_spherical(angle, -30.0, 4.0) for angle in angles], 0)
#     render_times_fixtime = np.array(
#         [0.0000, 0.0256, 0.0513, 0.0769, 0.1026, 0.1026, 0.1026, 0.1026, 0.2051,
#          0.2051, 0.2051, 0.2051, 0.3077, 0.3077, 0.3077, 0.3077, 0.4103, 0.4103,
#          0.4103, 0.4103, 0.5128, 0.5128, 0.5128, 0.5128, 0.6154, 0.6154, 0.6154,
#          0.6154, 0.7179, 0.7179, 0.7179, 0.7179, 0.8205, 0.8205, 0.8205, 0.8205,
#          0.9231, 0.9231, 0.9231, 0.9231])
#
#
#     return render_poses, render_times, render_poses_fixview, render_times_fixview, render_poses_fixtime, render_times_fixtime
