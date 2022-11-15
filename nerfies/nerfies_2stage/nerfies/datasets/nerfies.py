# Copyright 2021 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Casual Volumetric Capture datasets."""
import json
from typing import List, Tuple
import os
from absl import logging
import cv2
import numpy as np
import torch
import imageio

from nerfies import gpath
from nerfies import types
from nerfies import utils
from nerfies import image_utils
from nerfies.datasets import core
from nerfies.getRender import getRender

# trans_t = lambda t : torch.Tensor([
#     [1,0,0,0],
#     [0,1,0,0],
#     [0,0,1,t],
#     [0,0,0,1]]).float()
#
# rot_phi = lambda phi : torch.Tensor([
#     [1,0,0,0],
#     [0,np.cos(phi),-np.sin(phi),0],
#     [0,np.sin(phi), np.cos(phi),0],
#     [0,0,0,1]]).float()
#
# rot_theta = lambda th : torch.Tensor([
#     [np.cos(th),0,-np.sin(th),0],
#     [0,1,0,0],
#     [np.sin(th),0, np.cos(th),0],
#     [0,0,0,1]]).float()
#
#
# def pose_spherical(theta, phi, radius):
#     c2w = trans_t(radius)
#     c2w = rot_phi(phi/180.*np.pi) @ c2w
#     c2w = rot_theta(theta/180.*np.pi) @ c2w
#     c2w = torch.Tensor(np.array([[-1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]])) @ c2w
#     return c2w

# def getRender():
#     N_perpose = 50
#     N_allpose = 0
#     angles_all = np.linspace(-180, 180, 10 + 1)
#     angles_all = np.linspace(-108, 252, 10 + 1)  # daisy robot
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
#     angles = np.linspace(-144, 144, 144 + 1)
#     angles = np.linspace(-72, 216, 144 + 1)  # robot
#     # thetas1 = np.linspace(-30, -90, int(len(angles) / 2)+1)
#     # thetas2 = np.linspace(-90, -30, int(len(angles) / 2)+1)
#     # thetas = np.concatenate((thetas1, thetas2))
#     # render_poses.append(torch.stack([pose_spherical(angle, theta, 3.0) for (angle, theta) in zip(angles, thetas)]))
#     render_poses.append(torch.stack([pose_spherical(angle, -30.0, 4.0) for angle in angles]))
#     render_times.append(torch.ones([len(angles)]) * 0.5)
#     N_allpose += len(angles)
#     # position -2
#     # angles = np.ones([30]) * angles_all[-2]
#     angles = np.linspace(angles_all[-2], angles_all[-1], 30)
#     render_poses.append(torch.stack([pose_spherical(angle, -30.0, 4.0) for angle in angles]))
#     render_times.append(torch.linspace(0.5, 0, len(angles)))
#     N_allpose += len(angles)
#     # # position -1
#     # angles = np.ones([N_perpose]) * angles_all[-1]
#     # render_poses.append(torch.stack([pose_spherical(angle, -30.0, 4.0) for angle in angles]))
#     # render_times.append(torch.linspace(0., 1., N_perpose))
#     # N_allpose += len(angles)
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
# 
#   # Fix video_version2
#     N_perpose = 50
#     N_allpose = 0
#     angles_all = np.linspace(-180, 180, 10 + 1)
#     angles_all = np.linspace(-108, 252, 10 + 1)  # daisy robot
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
#     angles = np.linspace(-144, 144, 144 + 1)
#     angles = np.linspace(-72, 216, 144 + 1)  # robot
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
#     angles = np.linspace(angles_all[-2], angles_all[-1], 30)
#     render_poses_fixview.append(torch.stack([pose_spherical(angle, -30.0, 4.0) for angle in angles]))
#     render_times_fixview.append(torch.linspace(0.5, 0, len(angles)))
#     N_allpose += len(angles)
#     # # position -1
#     # angles = np.ones([N_perpose]) * angles_all[-1]
#     # render_poses_fixview.append(torch.stack([pose_spherical(angle, -30.0, 4.0) for angle in angles]))
#     # render_times_fixview.append(torch.linspace(0., 1., N_perpose))
#     # N_allpose += len(angles)
# 
#     render_poses_fixview = torch.cat(render_poses_fixview).reshape([N_allpose, 4, 4])
#     render_times_fixview = torch.cat(render_times_fixview).reshape([N_allpose])
# 
#   #     angles = np.ones([40]) * (-180)
#   #     # angles = np.array(
#   #     # [-180., -180.,  -180., -180., -180., -180., -180., -180., -180.,
#   #     # -180.,  -180.,  -180., -180., -180., -180., -180., -180., -180.,
#   #     # -180.,  -180.,  -180., -180., -180., -180., -180., -180., -180.,
#   #     # -180.,  -180.,  -180., -180., -180., -180., -180., -180., -180.,
#   #     # -180.,  -180.,  -180., -180.])
#   #     render_poses_fixview = torch.stack([pose_spherical(angle, -30.0, 4.0) for angle in angles], 0)
#   # render_times_fixview = torch.linspace(0., 1., render_poses_fixview.shape[0])
# 
#   # Fix time and change view
#     # import pdb
#     # pdb.set_trace()
#     angles = np.array(
#       [-180., -171., -162., -153., -144., -135., -126., -117., -108.,
#        -99., -90., -81., -72., -63., -54., -45., -36., -27.,
#        -18., -9., 0., 9., 18., 27., 36., 45., 54.,
#        63., 72., 81., 90., 99., 108., 117., 126., 135.,
#        144., 153., 162., 171.])
#     # phis = np.array(
#     # [-180., -180.,  -180., -180., -180., -180., -180., -180., -180.,
#     # -180.,  -180.,  -180., -180., -180., -180., -180., -180., -180.,
#     # -180.,  -180.,  -180., -180., -180., -180., -180., -180., -180.,
#     # -180.,  -180.,  -180., -180., -180., -180., -180., -180., -180.,
#     # -180.,  -180.,  -180., -180.])
#     render_poses_fixtime = torch.stack([pose_spherical(angle, -30.0, 4.0) for angle in angles], 0)
#     render_times_fixtime = np.array(
#       [0.0000, 0.0256, 0.0513, 0.0769, 0.1026, 0.1026, 0.1026, 0.1026, 0.2051,
#        0.2051, 0.2051, 0.2051, 0.3077, 0.3077, 0.3077, 0.3077, 0.4103, 0.4103,
#        0.4103, 0.4103, 0.5128, 0.5128, 0.5128, 0.5128, 0.6154, 0.6154, 0.6154,
#        0.6154, 0.7179, 0.7179, 0.7179, 0.7179, 0.8205, 0.8205, 0.8205, 0.8205,
#        0.9231, 0.9231, 0.9231, 0.9231])
# 
#     return  render_poses,render_times,render_poses_fixview, render_times_fixview, render_poses_fixtime, render_times_fixtime

    

def load_scene_info(
    data_dir: types.PathType) -> Tuple[np.ndarray, float, float, float]:
  """Loads the scene scale from scene_scale.npy.

  Args:
    data_dir: the path to the dataset.
    scene_center: the center of the scene (unscaled coordinates).
    scene_scale: the scale of the scene.
    near: the near plane of the scene (scaled coordinates).
    far: the far plane of the scene (scaled coordinates).

  Raises:
    ValueError if scene_scale.npy does not exist.
  """
  # scene_json_path = gpath.GPath(data_dir, 'scene.json')
  # with scene_json_path.open('r') as f:
  #   scene_json = json.load(f)
  #
  # scene_center = np.array(scene_json['center'])
  # scene_scale = scene_json['scale']
  # near = scene_json['near']
  # far = scene_json['far']
  far = 6
  near =2
  scene_scale = 1
  scene_center = None

  return scene_center, scene_scale, near, far


def _load_image(path: types.PathType) -> np.ndarray:
  path = gpath.GPath(path)
  with path.open('rb') as f:
    raw_im = np.asarray(bytearray(f.read()), dtype=np.uint8)
    path = str(path)
    img = imageio.imread(path)
    img = (np.array(img) / 255.).astype(np.float32)
    img = img[...,:3]
    # img = img[..., :3] * img[..., -1:] + (1. - img[..., -1:])

    # image = cv2.imdecode(raw_im, cv2.IMREAD_COLOR)[:, :, ::-1]  # BGR -> RGB
    # image = np.asarray(image).astype(np.float32) / 255.0
    # print(image.shape)
    # print(img.shape)
    # image_utils.save_image(
    #   f"/home/jiawei/weijia/literature/ourModels/nerfies_1stage/lego_4views_no_floor_final/renders/gt.png",
    #   image_utils.image_to_uint8(img))
    # input()
    return img


def _load_dataset_ids(data_dir: types.PathType) -> Tuple[List[str], List[str]]:
  """Loads dataset IDs."""
  # dataset_json_path = gpath.GPath(data_dir, 'dataset.json')
  # logging.info('*** Loading dataset IDs from %s', dataset_json_path)
  # with dataset_json_path.open('r') as f:
  #   dataset_json = json.load(f)
  #   train_ids = dataset_json['train_ids']
  #   val_ids = dataset_json['val_ids']
  #
  # train_ids = [str(i) for i in train_ids]
  # val_ids = [str(i) for i in val_ids]
  dataset_json_path = gpath.GPath(data_dir, 'transforms_train.json')
  # print(dataset_json_path)
  train_ids = []
  with dataset_json_path.open('r') as f:
    dataset_json = json.load(f)
    for frame in dataset_json['frames']:
      train_ids.append(frame['file_path'])

  dataset_json_path = gpath.GPath(data_dir, 'transforms_test.json')
  val_ids = []
  with dataset_json_path.open('r') as f:
    dataset_json = json.load(f)
    for frame in dataset_json['frames']:
      val_ids.append(frame['file_path'])

  return train_ids, val_ids

  # return train_ids, val_ids


class NerfiesDataSource(core.DataSource):
  """Data loader for videos."""

  def __init__(
      self,
      data_dir,
      image_scale: int,
      shuffle_pixels=False,
      camera_type='json',
      test_camera_trajectory='orbit-extreme',
      **kwargs):
    if data_dir !='':
      self.data_dir = gpath.GPath(data_dir)
      # Load IDs from JSON if it exists. This is useful since COLMAP fails on
      # some images so this gives us the ability to skip invalid images.
      train_ids, val_ids = _load_dataset_ids(self.data_dir)
      self.scene_center, self.scene_scale, self._near, self._far = \
        load_scene_info(self.data_dir)
      self.test_camera_trajectory = test_camera_trajectory

      self.image_scale = image_scale
      self.shuffle_pixels = shuffle_pixels

      # self.rgb_dir = gpath.GPath(data_dir, 'rgb', f'{image_scale}x')
      # self.depth_dir = gpath.GPath(data_dir, 'depth', f'{image_scale}x')
      # self.camera_type = camera_type
      # self.camera_dir = gpath.GPath(data_dir, 'camera')

      self.rgb_dir = gpath.GPath(data_dir)
      self.depth_dir = gpath.GPath(data_dir, 'depth', f'{image_scale}x')
      self.camera_type = camera_type
      self.camera_dir = gpath.GPath(data_dir)

      metadata_path = self.data_dir / 'metadata.json'
      self.metadata_dict = None
      if metadata_path.exists():
        with metadata_path.open('r') as f:
          self.metadata_dict = json.load(f)
    else:
      render_poses, render_times, render_poses_fixview, render_times_fixview, render_poses_fixtime, render_times_fixtime = getRender()
      render_times_fixview = render_times_fixview*50
      # render_times_fixview = render_times * 50
      self.rgb_dir = None
      self.depth_dir = None
      val_ids = [i for i in range(len(render_poses_fixview))]
      train_ids = val_ids
      self.camera_type = camera_type
      self.camera_dir = None
      self.image_scale = 1
      self.metadata_dict = {}
      for i in range(len(render_poses_fixview)):
          self.metadata_dict[i] = {}
          self.metadata_dict[i]['time_id'] = int(render_times_fixview[i])
          self.metadata_dict[i]['warp_id'] = int(render_times_fixview[i])
          self.metadata_dict[i]['appearance_id'] = int(render_times_fixview[i])
          self.metadata_dict[i]['camera_id'] = 0
    self.scene_center, self.scene_scale, self._near, self._far = \
      load_scene_info('')
    super().__init__(train_ids=train_ids, val_ids=val_ids,
                     **kwargs)



  @property
  def near(self):
    return self._near

  @property
  def far(self):
    return self._far

  @property
  def camera_ext(self):
    if self.camera_type == 'json':
      return '.json'

    raise ValueError(f'Unknown camera_type {self.camera_type}')

  def get_rgb_path(self, item_id):
    return self.rgb_dir / f'{item_id}.png'
    # return self.rgb_dir / f'{item_id}'

  def load_rgb(self, item_id):
    if self.rgb_dir is not None:
        return _load_image(self.rgb_dir / f'{item_id}.png')
    else:
        return None
    # return _load_image(self.rgb_dir / f'{item_id}')

  def load_camera(self, item_id, scale_factor=1.0):
    if isinstance(item_id, gpath.GPath):
      camera_path = item_id
    else:
      if self.camera_type == 'json':
        camera_path = self.camera_dir
        #camera_path = self.camera_dir / f'{item_id}{self.camera_ext}'
      else:
        raise ValueError(f'Unknown camera type {self.camera_type!r}.')

    return core.load_camera(camera_path,
                            item_id,
                            scale_factor=scale_factor / self.image_scale,
                            scene_center=self.scene_center,
                            scene_scale=self.scene_scale)

  def glob_cameras(self, path):
    path = gpath.GPath(path)
    return sorted(path.glob(f'*{self.camera_ext}'))

  def load_test_cameras(self, count=None):
    camera_dir = (self.data_dir / 'camera-paths' / self.test_camera_trajectory)
    if not camera_dir.exists():
      logging.warning('test camera path does not exist: %s', str(camera_dir))
      return []
    camera_paths = sorted(camera_dir.glob(f'*{self.camera_ext}'))
    if count is not None:
      stride = max(1, len(camera_paths) // count)
      camera_paths = camera_paths[::stride]
    cameras = utils.parallel_map(self.load_camera, camera_paths)
    return cameras

  def load_points(self, shuffle=False):
    with (self.data_dir / 'points.npy').open('rb') as f:
      points = np.load(f)
    points = (points - self.scene_center) * self.scene_scale
    points = points.astype(np.float32)
    if shuffle:
      logging.info('Shuffling points.')
      shuffled_inds = self.rng.permutation(len(points))
      points = points[shuffled_inds]
    logging.info('Loaded %d points.', len(points))
    return points

  def get_appearance_id(self, item_id):
    return self.metadata_dict[item_id]['appearance_id']

  def get_camera_id(self, item_id):
    return self.metadata_dict[item_id]['camera_id']

  def get_warp_id(self, item_id):
    return self.metadata_dict[item_id]['warp_id']

  def get_time_id(self, item_id):
    if 'time_id' in self.metadata_dict[item_id]:
      return self.metadata_dict[item_id]['time_id']
    else:
      # Fallback for older datasets.
      return self.metadata_dict[item_id]['warp_id']
