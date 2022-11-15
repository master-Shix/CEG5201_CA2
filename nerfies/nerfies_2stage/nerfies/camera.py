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

"""Class for handling cameras."""
import copy
import json
import torch
from typing import Tuple, Union, Optional

import numpy as np

from nerfies import gpath
from nerfies import types
from nerfies.getRender import getRender
import cv2
#
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

#
# def getRender():
#     N_perpose = 50
#     N_allpose = 0
#     # angles_all = np.linspace(-180, 180, 10 + 1)
#     angles_all = np.linspace(-108, 252, 10 + 1) #daisy robot
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
#     # angles = np.linspace(-144, 144, 144 + 1)
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
#     # Fix video_version2
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
#     #     angles = np.ones([40]) * (-180)
#     #     # angles = np.array(
#     #     # [-180., -180.,  -180., -180., -180., -180., -180., -180., -180.,
#     #     # -180.,  -180.,  -180., -180., -180., -180., -180., -180., -180.,
#     #     # -180.,  -180.,  -180., -180., -180., -180., -180., -180., -180.,
#     #     # -180.,  -180.,  -180., -180., -180., -180., -180., -180., -180.,
#     #     # -180.,  -180.,  -180., -180.])
#     #     render_poses_fixview = torch.stack([pose_spherical(angle, -30.0, 4.0) for angle in angles], 0)
#     # render_times_fixview = torch.linspace(0., 1., render_poses_fixview.shape[0])
#
#     # Fix time and change view
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
#     return render_poses, render_times, render_poses_fixview, render_times_fixview, render_poses_fixtime, render_times_fixtime


def _compute_residual_and_jacobian(
    x: np.ndarray,
    y: np.ndarray,
    xd: np.ndarray,
    yd: np.ndarray,
    k1: float = 0.0,
    k2: float = 0.0,
    k3: float = 0.0,
    p1: float = 0.0,
    p2: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray,
           np.ndarray]:
  """Auxiliary function of radial_and_tangential_undistort()."""
  # let r(x, y) = x^2 + y^2;
  #     d(x, y) = 1 + k1 * r(x, y) + k2 * r(x, y) ^2 + k3 * r(x, y)^3;
  r = x * x + y * y
  d = 1.0 + r * (k1 + r * (k2 + k3 * r))

  # The perfect projection is:
  # xd = x * d(x, y) + 2 * p1 * x * y + p2 * (r(x, y) + 2 * x^2);
  # yd = y * d(x, y) + 2 * p2 * x * y + p1 * (r(x, y) + 2 * y^2);
  #
  # Let's define
  #
  # fx(x, y) = x * d(x, y) + 2 * p1 * x * y + p2 * (r(x, y) + 2 * x^2) - xd;
  # fy(x, y) = y * d(x, y) + 2 * p2 * x * y + p1 * (r(x, y) + 2 * y^2) - yd;
  #
  # We are looking for a solution that satisfies
  # fx(x, y) = fy(x, y) = 0;
  fx = d * x + 2 * p1 * x * y + p2 * (r + 2 * x * x) - xd
  fy = d * y + 2 * p2 * x * y + p1 * (r + 2 * y * y) - yd

  # Compute derivative of d over [x, y]
  d_r = (k1 + r * (2.0 * k2 + 3.0 * k3 * r))
  d_x = 2.0 * x * d_r
  d_y = 2.0 * y * d_r

  # Compute derivative of fx over x and y.
  fx_x = d + d_x * x + 2.0 * p1 * y + 6.0 * p2 * x
  fx_y = d_y * x + 2.0 * p1 * x + 2.0 * p2 * y

  # Compute derivative of fy over x and y.
  fy_x = d_x * y + 2.0 * p2 * y + 2.0 * p1 * x
  fy_y = d + d_y * y + 2.0 * p2 * x + 6.0 * p1 * y

  return fx, fy, fx_x, fx_y, fy_x, fy_y


def _radial_and_tangential_undistort(
    xd: np.ndarray,
    yd: np.ndarray,
    k1: float = 0,
    k2: float = 0,
    k3: float = 0,
    p1: float = 0,
    p2: float = 0,
    eps: float = 1e-9,
    max_iterations=10) -> Tuple[np.ndarray, np.ndarray]:
  """Computes undistorted (x, y) from (xd, yd)."""
  # Initialize from the distorted point.
  x = xd.copy()
  y = yd.copy()

  for _ in range(max_iterations):
    fx, fy, fx_x, fx_y, fy_x, fy_y = _compute_residual_and_jacobian(
        x=x, y=y, xd=xd, yd=yd, k1=k1, k2=k2, k3=k3, p1=p1, p2=p2)
    denominator = fy_x * fx_y - fx_x * fy_y
    x_numerator = fx * fy_y - fy * fx_y
    y_numerator = fy * fx_x - fx * fy_x
    step_x = np.where(
        np.abs(denominator) > eps, x_numerator / denominator,
        np.zeros_like(denominator))
    step_y = np.where(
        np.abs(denominator) > eps, y_numerator / denominator,
        np.zeros_like(denominator))

    x = x + step_x
    y = y + step_y

  return x, y


class Camera:
  """Class to handle camera geometry."""

  def __init__(self,
               orientation: np.ndarray,
               position: np.ndarray,
               focal_length: Union[np.ndarray, float],
               principal_point: np.ndarray,
               image_size: np.ndarray,
               skew: Union[np.ndarray, float] = 0.0,
               pixel_aspect_ratio: Union[np.ndarray, float] = 1.0,
               radial_distortion: Optional[np.ndarray] = None,
               tangential_distortion: Optional[np.ndarray] = None,
               dtype=np.float32):
    """Constructor for camera class."""
    if radial_distortion is None:
      radial_distortion = np.array([0.0, 0.0, 0.0], dtype)
    if tangential_distortion is None:
      tangential_distortion = np.array([0.0, 0.0], dtype)

    self.orientation = np.array(orientation, dtype)
    self.position = np.array(position, dtype)
    self.focal_length = np.array(focal_length, dtype)
    self.principal_point = np.array(principal_point, dtype)
    self.skew = np.array(skew, dtype)
    self.pixel_aspect_ratio = np.array(pixel_aspect_ratio, dtype)
    self.radial_distortion = np.array(radial_distortion, dtype)
    self.tangential_distortion = np.array(tangential_distortion, dtype)
    self.image_size = np.array(image_size, np.uint32)
    self.dtype = dtype


  @classmethod
  def from_json(cls, path: types.PathType,item_id):
    """Loads a JSON camera into memory."""
    transf = np.array([
        [1, 0, 0, ],
        [0, -1, 0],
        [0, 0, -1],
    ])
    camera = {}
    if path is not None:
        p = gpath.GPath(path, "transforms_train.json")
        with p.open('r') as f:
            dataset_json = json.load(f)
            camera_angle_x = dataset_json['camera_angle_x']
            for frame in dataset_json['frames']:
                file_path = frame['file_path']
                if (file_path == item_id):
                    poses = frame['transform_matrix']
                    poses = np.array(poses)
                    R = poses[0:3, 0:3]
                    t = poses[0:3, 3]
                    R = transf @ R.T
                    camera["orientation"] = R.tolist()
                    camera["position"] = t.tolist()


        p = gpath.GPath(path, "transforms_test.json")
        with p.open('r') as f:
            dataset_json = json.load(f)
            camera_angle_x = dataset_json['camera_angle_x']
            for frame in dataset_json['frames']:
                file_path = frame['file_path']
                if (file_path == item_id):
                    poses = frame['transform_matrix']
                    poses = np.array(poses)
                    R = poses[0:3, 0:3]
                    t = poses[0:3, 3]
                    R = transf @ R.T
                    camera["orientation"] = R.tolist()
                    camera["position"] = t.tolist()

        image = path / f'{item_id}.png'
        # image = path / f'{item_id}'
        with image.open('rb') as f:
            raw_im = np.asarray(bytearray(f.read()), dtype=np.uint8)
            img = cv2.imdecode(raw_im, cv2.IMREAD_COLOR)[:, :, ::-1]  # BGR -> RGB
        H, W = img.shape[:2]
        focal = .5 * W / np.tan(.5 * camera_angle_x)
        camera["focal_length"] = focal

    else:
        render_poses, render_times,render_poses_fixview, render_times_fixview, render_poses_fixtime, render_times_fixtime =getRender()
        poses = render_poses_fixview[item_id]
        # poses = render_poses[item_id]
        poses = np.array(poses)
        R = poses[0:3, 0:3]
        t = poses[0:3, 3]
        # W=960
        # H = 540
        W = H = 800
        R = transf @ R.T
        camera["orientation"] = R.tolist()
        camera["position"] = t.tolist()
        camera_angle_x = 0.6911112070083618
        # camera_angle_x = 1.6161922301457423
        camera['focal_length'] = .5 * W / np.tan(.5 * camera_angle_x)




    return cls(
        orientation=np.asarray(camera['orientation']),
        position=np.asarray(camera['position']),
        focal_length=camera['focal_length'],
        principal_point=np.array([W/2, H/2]),
        skew=0.0,
        pixel_aspect_ratio=1,
        radial_distortion=None,
        tangential_distortion=None,
        image_size=np.asarray([W, H]),
    )

  def to_json(self):
    return {
        k: (v.tolist() if hasattr(v, 'tolist') else v)
        for k, v in self.get_parameters().items()
    }

  def get_parameters(self):
    return {
        'orientation': self.orientation,
        'position': self.position,
        'focal_length': self.focal_length,
        'principal_point': self.principal_point,
        'skew': self.skew,
        'pixel_aspect_ratio': self.pixel_aspect_ratio,
        'radial_distortion': self.radial_distortion,
        'tangential_distortion': self.tangential_distortion,
        'image_size': self.image_size,
    }

  @property
  def scale_factor_x(self):
    return self.focal_length

  @property
  def scale_factor_y(self):
    return self.focal_length * self.pixel_aspect_ratio

  @property
  def principal_point_x(self):
    return self.principal_point[0]

  @property
  def principal_point_y(self):
    return self.principal_point[1]

  @property
  def has_tangential_distortion(self):
    return any(self.tangential_distortion != 0.0)

  @property
  def has_radial_distortion(self):
    return any(self.radial_distortion != 0.0)

  @property
  def image_size_y(self):
    return self.image_size[1]

  @property
  def image_size_x(self):
    return self.image_size[0]

  @property
  def image_shape(self):
    return self.image_size_y, self.image_size_x

  @property
  def optical_axis(self):
    return self.orientation[2, :]

  @property
  def translation(self):
    return -np.matmul(self.orientation, self.position)

  def pixel_to_local_rays(self, pixels: np.ndarray):
    """Returns the local ray directions for the provided pixels."""
    y = ((pixels[..., 1] - self.principal_point_y) / self.scale_factor_y)
    x = ((pixels[..., 0] - self.principal_point_x - y * self.skew) /
         self.scale_factor_x)

    if self.has_radial_distortion or self.has_tangential_distortion:
      x, y = _radial_and_tangential_undistort(
          x,
          y,
          k1=self.radial_distortion[0],
          k2=self.radial_distortion[1],
          k3=self.radial_distortion[2],
          p1=self.tangential_distortion[0],
          p2=self.tangential_distortion[1])

    dirs = np.stack([x, y, np.ones_like(x)], axis=-1)
    return dirs / np.linalg.norm(dirs, axis=-1, keepdims=True)

  def pixels_to_rays(self, pixels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Returns the rays for the provided pixels.

    Args:
      pixels: [A1, ..., An, 2] tensor or np.array containing 2d pixel positions.

    Returns:
        An array containing the normalized ray directions in world coordinates.
    """
    if pixels.shape[-1] != 2:
      raise ValueError('The last dimension of pixels must be 2.')
    if pixels.dtype != self.dtype:
      raise ValueError(f'pixels dtype ({pixels.dtype!r}) must match camera '
                       f'dtype ({self.dtype!r})')

    batch_shape = pixels.shape[:-1]
    pixels = np.reshape(pixels, (-1, 2))

    local_rays_dir = self.pixel_to_local_rays(pixels)
    rays_dir = np.matmul(self.orientation.T, local_rays_dir[..., np.newaxis])
    rays_dir = np.squeeze(rays_dir, axis=-1)

    # Normalize rays.
    rays_dir /= np.linalg.norm(rays_dir, axis=-1, keepdims=True)
    rays_dir = rays_dir.reshape((*batch_shape, 3))
    return rays_dir

  def pixels_to_points(self, pixels: np.ndarray, depth: np.ndarray):
    rays_through_pixels = self.pixels_to_rays(pixels)
    cosa = np.matmul(rays_through_pixels, self.optical_axis)
    points = (
        rays_through_pixels * depth[..., np.newaxis] / cosa[..., np.newaxis] +
        self.position)
    return points

  def points_to_local_points(self, points: np.ndarray):
    translated_points = points - self.position
    local_points = (np.matmul(self.orientation, translated_points.T)).T
    return local_points

  def project(self, points: np.ndarray):
    """Projects a 3D point (x,y,z) to a pixel position (x,y)."""
    batch_shape = points.shape[:-1]
    points = points.reshape((-1, 3))
    local_points = self.points_to_local_points(points)

    # Get normalized local pixel positions.
    x = local_points[..., 0] / local_points[..., 2]
    y = local_points[..., 1] / local_points[..., 2]
    r2 = x**2 + y**2

    # Apply radial distortion.
    distortion = 1.0 + r2 * (
        self.radial_distortion[0] + r2 *
        (self.radial_distortion[1] + self.radial_distortion[2] * r2))

    # Apply tangential distortion.
    x_times_y = x * y
    x = (
        x * distortion + 2.0 * self.tangential_distortion[0] * x_times_y +
        self.tangential_distortion[1] * (r2 + 2.0 * x**2))
    y = (
        y * distortion + 2.0 * self.tangential_distortion[1] * x_times_y +
        self.tangential_distortion[0] * (r2 + 2.0 * y**2))

    # Map the distorted ray to the image plane and return the depth.
    pixel_x = self.focal_length * x + self.skew * y + self.principal_point_x
    pixel_y = (self.focal_length * self.pixel_aspect_ratio * y
               + self.principal_point_y)

    pixels = np.stack([pixel_x, pixel_y], axis=-1)
    return pixels.reshape((*batch_shape, 2))

  def get_pixel_centers(self):
    """Returns the pixel centers."""
    xx, yy = np.meshgrid(np.arange(self.image_size_x, dtype=self.dtype),
                         np.arange(self.image_size_y, dtype=self.dtype))
    return np.stack([xx, yy], axis=-1) + 0.5

  def scale(self, scale: float):
    """Scales the camera."""
    if scale <= 0:
      raise ValueError('scale needs to be positive.')

    new_camera = Camera(
        orientation=self.orientation.copy(),
        position=self.position.copy(),
        focal_length=self.focal_length * scale,
        principal_point=self.principal_point.copy() * scale,
        skew=self.skew,
        pixel_aspect_ratio=self.pixel_aspect_ratio,
        radial_distortion=self.radial_distortion.copy(),
        tangential_distortion=self.tangential_distortion.copy(),
        image_size=np.array((int(round(self.image_size[0] * scale)),
                             int(round(self.image_size[1] * scale)))),
    )
    return new_camera

  def look_at(self, position, look_at, up, eps=1e-6):
    """Creates a copy of the camera which looks at a given point.

    Copies the provided vision_sfm camera and returns a new camera that is
    positioned at `camera_position` while looking at `look_at_position`.
    Camera intrinsics are copied by this method. A common value for the
    up_vector is (0, 1, 0).

    Args:
      position: A (3,) numpy array representing the position of the camera.
      look_at: A (3,) numpy array representing the location the camera looks at.
      up: A (3,) numpy array representing the up direction, whose projection is
        parallel to the y-axis of the image plane.
      eps: a small number to prevent divides by zero.

    Returns:
      A new camera that is copied from the original but is positioned and
        looks at the provided coordinates.

    Raises:
      ValueError: If the camera position and look at position are very close
        to each other or if the up-vector is parallel to the requested optical
        axis.
    """

    look_at_camera = self.copy()
    optical_axis = look_at - position
    norm = np.linalg.norm(optical_axis)
    if norm < eps:
      raise ValueError('The camera center and look at position are too close.')
    optical_axis /= norm

    right_vector = np.cross(optical_axis, up)
    norm = np.linalg.norm(right_vector)
    if norm < eps:
      raise ValueError('The up-vector is parallel to the optical axis.')
    right_vector /= norm

    # The three directions here are orthogonal to each other and form a right
    # handed coordinate system.
    camera_rotation = np.identity(3)
    camera_rotation[0, :] = right_vector
    camera_rotation[1, :] = np.cross(optical_axis, right_vector)
    camera_rotation[2, :] = optical_axis

    look_at_camera.position = position
    look_at_camera.orientation = camera_rotation
    return look_at_camera

  def crop_image_domain(
      self, left: int = 0, right: int = 0, top: int = 0, bottom: int = 0):
    """Returns a copy of the camera with adjusted image bounds.

    Args:
      left: number of pixels by which to reduce (or augment, if negative) the
        image domain at the associated boundary.
      right: likewise.
      top: likewise.
      bottom: likewise.

    The crop parameters may not cause the camera image domain dimensions to
    become non-positive.

    Returns:
      A camera with adjusted image dimensions.  The focal length is unchanged,
      and the principal point is updated to preserve the original principal
      axis.
    """

    crop_left_top = np.array([left, top])
    crop_right_bottom = np.array([right, bottom])
    new_resolution = self.image_size - crop_left_top - crop_right_bottom
    new_principal_point = self.principal_point - crop_left_top
    if np.any(new_resolution <= 0):
      raise ValueError('Crop would result in non-positive image dimensions.')

    new_camera = self.copy()
    new_camera.image_size = np.array([int(new_resolution[0]),
                                      int(new_resolution[1])])
    new_camera.principal_point = np.array([new_principal_point[0],
                                           new_principal_point[1]])
    return new_camera

  def copy(self):
    return copy.deepcopy(self)
