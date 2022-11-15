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

"""Evaluation script for Nerf."""
import collections
import functools
import time
from typing import Any, Dict, Optional, Sequence

from absl import app
from absl import flags
from absl import logging
from flax import jax_utils
from flax import optim
from flax.metrics import tensorboard
from flax.training import checkpoints
import gin
import jax
import os
from jax import numpy as jnp
from jax import random
import tensorflow as tf
import numpy as np

from nerfies import configs
from nerfies import datasets
from nerfies import evaluation
from nerfies import gpath
from nerfies import image_utils
from nerfies import model_utils
from nerfies import models_allon as models
from nerfies import types
from nerfies import utils
from nerfies import visualization as viz
import utils as utils_pips
import imageio
to8b = lambda x : (255*np.clip(x,0,1)).astype(np.uint8)

flags.DEFINE_enum('mode', None, ['jax_cpu', 'jax_gpu', 'jax_tpu'],
                  'Distributed strategy approach.')

flags.DEFINE_string('base_folder', "", 'where to store ckpts and logs')
flags.mark_flag_as_required('base_folder')
flags.DEFINE_string('data_dir', None, 'input data directory.')
flags.DEFINE_multi_string('gin_bindings', None, 'Gin parameter bindings.')
flags.DEFINE_multi_string('gin_configs', (), 'Gin config files.')
FLAGS = flags.FLAGS

jax.config.parse_flags_with_absl()


def compute_multiscale_ssim(image1: jnp.ndarray, image2: jnp.ndarray):
    """Compute the multiscale SSIM metric."""
    image1 = tf.convert_to_tensor(image1)
    image2 = tf.convert_to_tensor(image2)
    return tf.image.ssim_multiscale(image1, image2, max_val=1.0)


def process_batch(*,
                  batch: Dict[str, jnp.ndarray],
                  rng: types.PRNGKey,
                  state: model_utils.TrainState,
                  tag: str,
                  item_id: str,
                  step: int,
                  summary_writer: tensorboard.SummaryWriter,
                  render_fn: Any,
                  save_dir: Optional[gpath.GPath],
                  datasource: datasets.DataSource):
    """Process and plot a single batch."""
    # item_id = item_id.replace('/', '_')
    render = render_fn(state, batch, rng=rng)

    out = {}
    if jax.process_index() != 0:
        return out

    rgb = render['rgb']
    out['rgb'] = rgb
    acc = render['acc']
    depth_exp = render['depth']
    out['depth'] = depth_exp
    # depth_med = render['med_depth']
    # colorize_depth = functools.partial(viz.colorize,
    #                                    cmin=datasource.near,
    #                                    cmax=datasource.far,
    #                                    invert=True)

    # depth_exp_viz = colorize_depth(depth_exp)
    # depth_med_viz = colorize_depth(depth_med)
    # disp_exp_viz = viz.colorize(1.0 / depth_exp)
    # disp_med_viz = viz.colorize(1.0 / depth_med)
    # acc_viz = viz.colorize(acc, cmin=0.0, cmax=1.0)
    # if save_dir:
    #     save_dir.mkdir(parents=True, exist_ok=True)
    #     image_utils.save_image(save_dir / f'rgb_{item_id}.png',
    #                            image_utils.image_to_uint8(rgb))
    #     image_utils.save_image(save_dir / f'depth_expected_viz_{item_id}.png',
    #                            image_utils.image_to_uint8(depth_exp_viz))
    #     image_utils.save_depth(save_dir / f'depth_expected_{item_id}.png',
    #                            depth_exp)
    #     image_utils.save_image(save_dir / f'depth_median_viz_{item_id}.png',
    #                            image_utils.image_to_uint8(depth_med_viz))
    #     image_utils.save_depth(save_dir / f'depth_median_{item_id}.png',
    #                            depth_med)
    return out


def process_iterator(tag: str,
                     item_ids: Sequence[str],
                     iterator,
                     rng: types.PRNGKey,
                     state: model_utils.TrainState,
                     step: int,
                     render_fn: Any,
                     summary_writer: tensorboard.SummaryWriter,
                     save_dir: Optional[gpath.GPath],
                     datasource: datasets.DataSource):
    """Process a dataset iterator and compute metrics."""
    save_dir = save_dir / f'{step:08d}' / tag if save_dir else None
    meters = collections.defaultdict(utils.ValueMeter)
    rgbs = []
    depths = []
    n = 0
    for i, (item_id, batch) in enumerate(zip(item_ids, iterator)):
        # n+=1
        # if n==5:
        #     break
        logging.info('[%s:%d/%d] Processing %s ', tag, i + 1, len(item_ids), item_id)

        stats = process_batch(batch=batch,
                              rng=rng,
                              state=state,
                              tag=tag,
                              item_id=item_id,
                              step=step,
                              render_fn=render_fn,
                              summary_writer=summary_writer,
                              save_dir=save_dir,
                              datasource=datasource)
        rgb = np.array(stats['rgb'])
        rgbs.append(rgb)
        depth = np.array(stats['depth'])
        # print(depth)
        # input()
        depths.append(depth)

        if jax.process_index() == 0:
            for k, v in stats.items():
                meters[k].update(v)

    imageio.mimwrite(os.path.join(save_dir, 'video_rgb.mp4'), to8b(rgbs), fps=30, quality=8)
    # imageio.mimwrite(os.path.join(save_dir, 'video_depth.mp4'), to8b(1 - depths / np.max(depths)), fps=30, quality=8)
    imageio.mimwrite(os.path.join(save_dir, 'video_depth.mp4'), to8b(1- depths / np.max(depths)), fps=30, quality=8)
    input()

    if jax.process_index() == 0:
        for meter_name, meter in meters.items():
            summary_writer.scalar(tag=f'metrics-eval/{meter_name}/{tag}',
                                  value=meter.reduce('mean'),
                                  step=step)


def delete_old_renders(render_dir, max_renders):
    render_paths = sorted(render_dir.iterdir())
    paths_to_delete = render_paths[:-max_renders]
    for path in paths_to_delete:
        logging.info('Removing render directory %s', str(path))
        path.rmtree()


def main(argv):
    tf.config.experimental.set_visible_devices([], 'GPU')
    del argv
    logging.info('*** Starting experiment')
    gin_configs = FLAGS.gin_configs

    logging.info('*** Loading Gin configs from: %s', str(gin_configs))
    gin.parse_config_files_and_bindings(
        config_files=gin_configs,
        bindings=FLAGS.gin_bindings,
        skip_unknown=True)

    # Load configurations.
    exp_config = configs.ExperimentConfig()
    model_config = configs.ModelConfig(use_stratified_sampling=False)
    train_config = configs.TrainConfig()
    eval_config = configs.EvalConfig()

    # Get directory information.
    exp_dir = gpath.GPath(FLAGS.base_folder)
    if exp_config.subname:
        exp_dir = exp_dir / exp_config.subname
    logging.info('\texp_dir = %s', exp_dir)
    if not exp_dir.exists():
        exp_dir.mkdir(parents=True, exist_ok=True)

    summary_dir = exp_dir / 'summaries' / 'eval'
    logging.info('\tsummary_dir = %s', summary_dir)
    if not summary_dir.exists():
        summary_dir.mkdir(parents=True, exist_ok=True)

    renders_dir = exp_dir / 'renders'
    logging.info('\trenders_dir = %s', renders_dir)
    if not renders_dir.exists():
        renders_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_dir = exp_dir / 'checkpoints'
    logging.info('\tcheckpoint_dir = %s', checkpoint_dir)

    logging.info('Starting host %d. There are %d hosts : %s', jax.process_index(),
                 jax.process_count(), str(jax.process_index()))
    logging.info('Found %d accelerator devices: %s.', jax.local_device_count(),
                 str(jax.local_devices()))
    logging.info('Found %d total devices: %s.', jax.device_count(),
                 str(jax.devices()))

    rng = random.PRNGKey(20200823)

    devices_to_use = jax.local_devices()
    n_devices = len(
        devices_to_use) if devices_to_use else jax.local_device_count()

    datasource_spec = exp_config.datasource_spec
    if datasource_spec is None:
        datasource_spec = {
            'type': exp_config.datasource_type,
            'data_dir': FLAGS.data_dir,
        }
    logging.info('Creating datasource: %s', datasource_spec)
    datasource = datasets.from_config(
        datasource_spec,
        image_scale=exp_config.image_scale,
        use_appearance_id=model_config.use_appearance_metadata,
        use_camera_id=model_config.use_camera_metadata,
        use_warp_id=model_config.use_warp,
        use_time=True,
        random_seed=exp_config.random_seed,
        **exp_config.datasource_kwargs)

    # Get training IDs to evaluate.
    # train_eval_ids = utils.strided_subset(
    #     datasource.train_ids, eval_config.num_train_eval)
    # print(train_eval_ids)
    # input()
    # train_eval_iter = datasource.create_iterator2(train_eval_ids, batch_size=0)
    val_eval_ids = utils.strided_subset(
        datasource.val_ids, eval_config.num_val_eval)
    print(val_eval_ids)
    val_eval_iter = datasource.create_iterator2(val_eval_ids, batch_size=0)

    # test_cameras = datasource.load_test_cameras(count=eval_config.num_test_eval)
    # if test_cameras:
    #     test_dataset = datasource.create_cameras_dataset(test_cameras)
    #     test_eval_ids = [f'{x:03d}' for x in range(len(test_cameras))]
    #     test_eval_iter = datasets.iterator_from_dataset(test_dataset, batch_size=0)
    # else:
    #     test_eval_ids = None
    #     test_eval_iter = None

    rng, key = random.split(rng)
    params = {}
    model, params['model'] = models.construct_nerf(
        key,
        model_config,
        batch_size=eval_config.chunk,
        appearance_ids=datasource.appearance_ids,
        camera_ids=datasource.camera_ids,
        warp_ids=datasource.warp_ids,
        near=datasource.near,
        far=datasource.far,
        use_warp_jacobian=False,
        use_weights=False)

    optimizer_def = optim.Adam(0.0)
    optimizer = optimizer_def.create(params)
    init_state = model_utils.TrainState(optimizer=optimizer)
    del params

    def _model_fn(key_0, key_1, params, rays_dict, warp_extra):
        out = model.apply({'params': params},
                          rays_dict,
                          warp_extra=warp_extra,
                          rngs={
                              'coarse': key_0,
                              'fine': key_1
                          },
                          mutable=False)
        return jax.lax.all_gather(out, axis_name='batch')

    pmodel_fn = jax.pmap(
        # Note rng_keys are useless in eval mode since there's no randomness.
        _model_fn,
        in_axes=(0, 0, 0, 0, 0),  # Only distribute the data input.
        devices=devices_to_use,
        donate_argnums=(3,),  # Donate the 'rays' argument.
        axis_name='batch',
    )

    render_fn = functools.partial(evaluation.render_image,
                                  model_fn=pmodel_fn,
                                  device_count=n_devices,
                                  chunk=eval_config.chunk)

    last_step = 0
    summary_writer = tensorboard.SummaryWriter(str(summary_dir))

    while True:
        if not checkpoint_dir.exists():
            logging.info('No checkpoints yet.')
            time.sleep(10)
            continue

        state = checkpoints.restore_checkpoint(checkpoint_dir, init_state)
        state = jax_utils.replicate(state, devices=devices_to_use)
        step = int(state.optimizer.state.step[0])
        if step <= last_step:
            logging.info('No new checkpoints (%d <= %d).', step, last_step)
            time.sleep(10)
            continue

        save_dir = renders_dir if eval_config.save_output else None

        process_iterator(
            tag='val',
            item_ids=val_eval_ids,
            iterator=val_eval_iter,
            state=state,
            rng=rng,
            step=step,
            render_fn=render_fn,
            summary_writer=summary_writer,
            save_dir=save_dir,
            datasource=datasource)

        if save_dir:
            delete_old_renders(renders_dir, eval_config.max_render_checkpoints)

        if eval_config.eval_once:
            break
        if step >= train_config.max_steps:
            break
        last_step = step


if __name__ == '__main__':
    app.run(main)
