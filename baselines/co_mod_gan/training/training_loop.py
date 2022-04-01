# Copyright (c) 2019, NVIDIA Corporation. All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, visit
# https://nvlabs.github.io/stylegan2/license.html

"""Main training script."""

import numpy as np
import tensorflow as tf
import dnnlib
import dnnlib.tflib as tflib
from dnnlib.tflib.autosummary import autosummary

from training import dataset
from training import misc
from metrics import metric_base
import wandb
import torch
from torch.utils.data import DataLoader
from vae_helpers import rng_decorator
import os


@rng_decorator(0)
def log_images(args, model, training_set, viz_batch_processed, drange_net):
    def inpaint(reals, masks):
        batch_size = len(reals)
        labels = labels = np.zeros([batch_size, 0])
        latents = np.random.randn(batch_size, *model.input_shape[1:])
        fakes = model.run(latents, labels, reals, masks)
        return fakes

    masks = dataset.sample_mask(args,
                torch.zeros([args.num_images_visualize] + training_set.shape)).numpy()
    #masks = training_set.get_random_masks_np(args.num_images_visualize)
    reals = misc.adjust_dynamic_range(viz_batch_processed, training_set.dynamic_range, drange_net)
    log_dict = {}
    for idx in range(len(viz_batch_processed)):
        to_plot = [reals[idx] * masks[idx],
                   reals[idx]]
        reals_repeated = np.repeat(np.expand_dims(reals[idx], axis=0), args.num_samples_visualize, axis=0)
        masks_repeated = np.repeat(np.expand_dims(masks[idx], axis=0), args.num_samples_visualize, axis=0)
        fakes = inpaint(reals_repeated, masks_repeated)
        to_plot.extend(list(fakes))
        to_plot = [misc.convert_to_np_image(x, drange=drange_net) for x in to_plot]
        to_plot = np.concatenate(to_plot, axis=1)
        caption = f"Sample {idx}"
        log_dict.update({caption: wandb.Image(to_plot, caption=caption)})
    wandb.log(log_dict)


#----------------------------------------------------------------------------
# Just-in-time processing of training images before feeding them to the networks.

def process_reals(x, labels, lod, mirror_augment, drange_data, drange_net):
    with tf.name_scope('DynamicRange'):
        x = tf.cast(x, tf.float32)
        x = misc.adjust_dynamic_range(x, drange_data, drange_net)
    if mirror_augment:
        with tf.name_scope('MirrorAugment'):
            x = tf.where(tf.random_uniform([tf.shape(x)[0]]) < 0.5, x, tf.reverse(x, [3]))
    with tf.name_scope('FadeLOD'): # Smooth crossfade between consecutive levels-of-detail.
        s = tf.shape(x)
        y = tf.reshape(x, [-1, s[1], s[2]//2, 2, s[3]//2, 2])
        y = tf.reduce_mean(y, axis=[3, 5], keepdims=True)
        y = tf.tile(y, [1, 1, 1, 2, 1, 2])
        y = tf.reshape(y, [-1, s[1], s[2], s[3]])
        x = tflib.lerp(x, y, lod - tf.floor(lod))
    with tf.name_scope('UpscaleLOD'): # Upscale to match the expected input/output size of the networks.
        s = tf.shape(x)
        factor = tf.cast(2 ** tf.floor(lod), tf.int32)
        x = tf.reshape(x, [-1, s[1], s[2], 1, s[3], 1])
        x = tf.tile(x, [1, 1, 1, factor, 1, factor])
        x = tf.reshape(x, [-1, s[1], s[2] * factor, s[3] * factor])
    return x, labels

#----------------------------------------------------------------------------
# Evaluate time-varying training parameters.

def training_schedule(
    cur_nimg,
    training_set,
    lod_initial_resolution  = None,     # Image resolution used at the beginning.
    lod_training_kimg       = 600,      # Thousands of real images to show before doubling the resolution.
    lod_transition_kimg     = 600,      # Thousands of real images to show when fading in new layers.
    minibatch_size_base     = 32,       # Global minibatch size.
    minibatch_size_dict     = {},       # Resolution-specific overrides.
    minibatch_gpu_base      = 4,        # Number of samples processed at a time by one GPU.
    minibatch_gpu_dict      = {32: 16, 256: 8},       # Resolution-specific overrides.
    G_lrate_base            = 0.002,    # Learning rate for the generator.
    G_lrate_dict            = {},       # Resolution-specific overrides.
    D_lrate_base            = 0.002,    # Learning rate for the discriminator.
    D_lrate_dict            = {},       # Resolution-specific overrides.
    lrate_rampup_kimg       = 0,        # Duration of learning rate ramp-up.
    tick_kimg_base          = 8,        # Default interval of progress snapshots.
    tick_kimg_dict          = {}): # Resolution-specific overrides.
    # tick_kimg_base          = 4,        # Default interval of progress snapshots.
    # tick_kimg_dict          = {8:28, 16:24, 32:20, 64:16, 128:12, 256:8, 512:6, 1024:4}): # Resolution-specific overrides.

    # Initialize result dict.
    s = dnnlib.EasyDict()
    s.kimg = cur_nimg / 1000.0

    # Training phase.
    phase_dur = lod_training_kimg + lod_transition_kimg
    phase_idx = int(np.floor(s.kimg / phase_dur)) if phase_dur > 0 else 0
    phase_kimg = s.kimg - phase_idx * phase_dur

    # Level-of-detail and resolution.
    if lod_initial_resolution is None:
        s.lod = 0.0
    else:
        s.lod = training_set.resolution_log2
        s.lod -= np.floor(np.log2(lod_initial_resolution))
        s.lod -= phase_idx
        if lod_transition_kimg > 0:
            s.lod -= max(phase_kimg - lod_training_kimg, 0.0) / lod_transition_kimg
        s.lod = max(s.lod, 0.0)
    s.resolution = 2 ** (training_set.resolution_log2 - int(np.floor(s.lod)))

    # Minibatch size.
    s.minibatch_size = minibatch_size_dict.get(s.resolution, minibatch_size_base)
    s.minibatch_gpu = minibatch_gpu_dict.get(s.resolution, minibatch_gpu_base)

    # Learning rate.
    s.G_lrate = G_lrate_dict.get(s.resolution, G_lrate_base)
    s.D_lrate = D_lrate_dict.get(s.resolution, D_lrate_base)
    if lrate_rampup_kimg > 0:
        rampup = min(s.kimg / lrate_rampup_kimg, 1.0)
        s.G_lrate *= rampup
        s.D_lrate *= rampup

    # Other parameters.
    s.tick_kimg = tick_kimg_dict.get(s.resolution, tick_kimg_base)
    return s

#----------------------------------------------------------------------------
# Main training script.

def training_loop(
    args,
    G_args                  = {},       # Options for generator network.
    D_args                  = {},       # Options for discriminator network.
    G_opt_args              = {},       # Options for generator optimizer.
    D_opt_args              = {},       # Options for discriminator optimizer.
    G_loss_args             = {},       # Options for generator loss.
    D_loss_args             = {},       # Options for discriminator loss.
    dataset_args            = {},       # Options for dataset.load_dataset().
    sched_args              = {},       # Options for train.TrainingSchedule.
    grid_args               = {},       # Options for train.setup_snapshot_image_grid().
    metric_arg_list         = [],       # Options for MetricGroup.
    tf_config               = {},       # Options for tflib.init_tf().
    G_smoothing_kimg        = 10.0,     # Half-life of the running average of generator weights.
    minibatch_repeats       = 4,        # Number of minibatches to run before adjusting training parameters.
    lazy_regularization     = True,     # Perform regularization as a separate training step?
    G_reg_interval          = 4,        # How often the perform regularization for G? Ignored if lazy_regularization=False.
    D_reg_interval          = 16,       # How often the perform regularization for D? Ignored if lazy_regularization=False.
    reset_opt_for_new_lod   = True,     # Reset optimizer internal state (e.g. Adam moments) when new layers are introduced?
    drange_net              = [-1,1],   # Dynamic range used when feeding image data to the networks.
    network_snapshot_ticks  = 50,       # How often to save network snapshots? None = only save 'networks-final.pkl'.
    save_tf_graph           = False,    # Include full TensorFlow computation graph in the tfevents file?
    save_weight_histograms  = False,    # Include weight histograms in the tfevents file?
    resume_pkl              = None,     # Network pickle to resume training from, None = train from scratch.
    resume_kimg             = 0.0,      # Assumed training progress at the beginning. Affects reporting and training schedule.
    resume_time             = 0.0,      # Assumed wallclock time at the beginning. Affects reporting.
    resume_with_new_nets    = False):   # Construct new networks according to G_args and D_args before resuming training?,
    data_dir                = None if "data_dir" not in args.__dict__ else args.data_dir     # Directory to load datasets from.
    total_kimg              = 25000 if "total_kimg" not in args.__dict__ else args.total_kimg    # Total length of the training, measured in thousands of real images.
    mirror_augment          = False if "mirror_augment" not in args.__dict__ else args.mirror_augment    # Enable mirror augment?
    image_snapshot_ticks    = 50 if "image_snapshot_ticks" not in args.__dict__ else args.image_snapshot_ticks       # How often to save image snapshots? None = only save 'reals.png' and 'fakes-init.png'.

    # Initialize dnnlib and TensorFlow.
    tflib.init_tf(tf_config)
    num_gpus = dnnlib.submit_config.num_gpus

    # Load training set.
    training_set = dataset.NewDataset(args)
    viz_batch_processed = next(iter(DataLoader(training_set.valid_set,
                                batch_size=args.num_images_visualize)))[0].numpy()
    if args.save_image_grids:
        grid_size, grid_reals, grid_labels, grid_masks = misc.setup_snapshot_image_grid(training_set, **grid_args)
        img_path = dnnlib.make_run_dir_path('reals.png')
        misc.save_image_grid(grid_reals, img_path, drange=training_set.dynamic_range, grid_size=grid_size, pix2pix=training_set.pix2pix)
        grid_reals = misc.adjust_dynamic_range(grid_reals, training_set.dynamic_range, drange_net)

    # Construct or load networks.
    with tf.device('/gpu:0'):
        if resume_pkl is None or resume_with_new_nets:
            print('Constructing networks...')
            G = tflib.Network('G', num_channels=training_set.shape[0], resolution=training_set.shape[1], label_size=training_set.label_size,
                pix2pix=training_set.pix2pix, **G_args)
            D = tflib.Network('D', num_channels=training_set.shape[0], resolution=training_set.shape[1], label_size=training_set.label_size,
                pix2pix=training_set.pix2pix, **D_args)
            Gs = G.clone('Gs')
        if resume_pkl is not None:
            print('Loading networks from "%s"...' % resume_pkl)
            rG, rD, rGs = misc.load_pkl(resume_pkl)
            if resume_with_new_nets: G.copy_vars_from(rG); D.copy_vars_from(rD); Gs.copy_vars_from(rGs)
            else: G = rG; D = rD; Gs = rGs

    # Print layers and generate initial image snapshot.
    G.print_layers(); D.print_layers()
    sched = training_schedule(cur_nimg=total_kimg*1000, training_set=training_set, **sched_args)
    if args.save_image_grids:
        grid_latents = np.random.randn(np.prod(grid_size), *G.input_shape[1:])
        grid_fakes = Gs.run(grid_latents, grid_labels, grid_reals, grid_masks, minibatch_size=sched.minibatch_gpu)
        misc.save_image_grid(grid_fakes, dnnlib.make_run_dir_path('fakes_init.png'), drange=drange_net, grid_size=grid_size, pix2pix=training_set.pix2pix)

    # Setup training inputs.
    print('Building TensorFlow graph...')
    lod_in               = tf.convert_to_tensor(sched.lod)
    lrate_in             = tf.convert_to_tensor(sched.G_lrate)
    minibatch_size_in    = tf.convert_to_tensor(sched.minibatch_size)
    minibatch_gpu_in     = tf.convert_to_tensor(sched.minibatch_gpu)
    minibatch_multiplier = minibatch_size_in // (minibatch_gpu_in * num_gpus)
    Gs_beta              = 0.5 ** tf.div(tf.cast(minibatch_size_in, tf.float32), G_smoothing_kimg * 1000.0) if G_smoothing_kimg > 0.0 else 0.0

    # Setup optimizers.
    G_opt_args = dict(G_opt_args)
    D_opt_args = dict(D_opt_args)
    for _args, reg_interval in [(G_opt_args, G_reg_interval), (D_opt_args, D_reg_interval)]:
        _args['minibatch_multiplier'] = minibatch_multiplier
        _args['learning_rate'] = lrate_in
        if lazy_regularization:
            mb_ratio = reg_interval / (reg_interval + 1)
            _args['learning_rate'] *= mb_ratio
            if 'beta1' in _args: _args['beta1'] **= mb_ratio
            if 'beta2' in _args: _args['beta2'] **= mb_ratio
    G_opt = tflib.Optimizer(name='TrainG', **G_opt_args)
    D_opt = tflib.Optimizer(name='TrainD', **D_opt_args)
    G_reg_opt = tflib.Optimizer(name='RegG', share=G_opt, **G_opt_args)
    D_reg_opt = tflib.Optimizer(name='RegD', share=D_opt, **D_opt_args)

    reals_batch = tf.placeholder(training_set.dtype, name='reals', shape=[None] + training_set.shape)
    labels_batch = tf.placeholder(training_set.label_dtype, name='labels', shape=[None] + [training_set.label_size])
    masks_batch = tf.placeholder(tf.float32, name='reals', shape=[None, 1] + training_set.shape[1:])
    reals_gpu_list = tf.split(reals_batch, num_gpus, axis=0)
    labels_gpu_list = tf.split(labels_batch, num_gpus, axis=0)
    masks_gpu_list = tf.split(masks_batch, num_gpus, axis=0)

    # Build training graph for each GPU.
    data_fetch_ops = []
    for gpu in range(num_gpus):
        def auto_gpu(opr):
            if False:#opr.type in ['ExtractImagePatches', 'SparseSlice']:
                return '/cpu:0'
            else:
                return '/gpu:%d' % gpu
        
        with tf.name_scope('GPU%d' % gpu), tf.device(auto_gpu):

            # Create GPU-specific shadow copies of G and D.
            G_gpu = G if gpu == 0 else G.clone(G.name + '_shadow')
            D_gpu = D if gpu == 0 else D.clone(D.name + '_shadow')

            # Fetch training data via temporary variables.
            with tf.name_scope('DataFetch'):
                #index_range = [gpu * minibatch_gpu_in, (gpu+1) * minibatch_gpu_in]
                reals_gpu = reals_gpu_list[gpu]
                labels_gpu = labels_gpu_list[gpu]
                masks_gpu = masks_gpu_list[gpu]
                reals_write, labels_write = process_reals(reals_gpu, labels_gpu, lod_in, mirror_augment, training_set.dynamic_range, drange_net)
                masks_write = masks_gpu
                reals_var = tf.Variable(name='reals', trainable=False, initial_value=tf.zeros([minibatch_gpu_in] + training_set.shape))
                labels_var = tf.Variable(name='labels', trainable=False, initial_value=tf.zeros([minibatch_gpu_in, training_set.label_size]))
                masks_var = tf.Variable(name='masks', trainable=False, initial_value=tf.zeros([minibatch_gpu_in, 1] + training_set.shape[1:]))
                # reals_write = tf.concat([reals_write, reals_var[minibatch_gpu_in:]], axis=0)
                # labels_write = tf.concat([labels_write, labels_var[minibatch_gpu_in:]], axis=0)
                # masks_write = tf.concat([masks_write, masks_var[minibatch_gpu_in:]], axis=0)
                data_fetch_ops += [tf.assign(reals_var, reals_write)]
                data_fetch_ops += [tf.assign(labels_var, labels_write)]
                data_fetch_ops += [tf.assign(masks_var, masks_write)]
                reals_read = reals_var
                labels_read = labels_var
                masks_read = masks_var

            # Evaluate loss functions.
            lod_assign_ops = []
            if 'lod' in G_gpu.vars: lod_assign_ops += [tf.assign(G_gpu.vars['lod'], lod_in)]
            if 'lod' in D_gpu.vars: lod_assign_ops += [tf.assign(D_gpu.vars['lod'], lod_in)]
            with tf.control_dependencies(lod_assign_ops):
                with tf.name_scope('G_loss'):
                    G_loss, G_reg = dnnlib.util.call_func_by_name(G=G_gpu, D=D_gpu, opt=G_opt, training_set=training_set, minibatch_size=minibatch_gpu_in, reals=reals_read, masks=masks_read, **G_loss_args)
                with tf.name_scope('D_loss'):
                    D_loss, D_reg = dnnlib.util.call_func_by_name(G=G_gpu, D=D_gpu, opt=D_opt, training_set=training_set, minibatch_size=minibatch_gpu_in, reals=reals_read, labels=labels_read, masks=masks_read, **D_loss_args)

            # Register gradients.
            if not lazy_regularization:
                if G_reg is not None: G_loss += G_reg
                if D_reg is not None: D_loss += D_reg
            else:
                if G_reg is not None: G_reg_opt.register_gradients(tf.reduce_mean(G_reg * G_reg_interval), G_gpu.trainables)
                if D_reg is not None: D_reg_opt.register_gradients(tf.reduce_mean(D_reg * D_reg_interval), D_gpu.trainables)
            G_opt.register_gradients(tf.reduce_mean(G_loss), G_gpu.trainables)
            D_opt.register_gradients(tf.reduce_mean(D_loss), D_gpu.trainables)

    # Setup training ops.
    data_fetch_op = tf.group(*data_fetch_ops)
    G_train_op = G_opt.apply_updates()
    D_train_op = D_opt.apply_updates()
    G_reg_op = G_reg_opt.apply_updates(allow_no_op=True)
    D_reg_op = D_reg_opt.apply_updates(allow_no_op=True)
    Gs_update_op = Gs.setup_as_moving_average_of(G, beta=Gs_beta)

    # Finalize graph.
    with tf.device('/gpu:0'):
        try:
            peak_gpu_mem_op = tf.contrib.memory_stats.MaxBytesInUse()
        except tf.errors.NotFoundError:
            peak_gpu_mem_op = tf.constant(0)
    tflib.init_uninitialized_vars()

    print('Initializing logs...')
    summary_log = tf.summary.FileWriter(dnnlib.make_run_dir_path())
    if save_tf_graph:
        summary_log.add_graph(tf.get_default_graph())
    if save_weight_histograms:
        G.setup_weight_histograms(); D.setup_weight_histograms()
    metrics = metric_base.MetricGroup(metric_arg_list)

    print('Training for %d kimg...\n' % total_kimg)
    dnnlib.RunContext.get().update('', cur_epoch=resume_kimg, max_epoch=total_kimg)
    maintenance_time = dnnlib.RunContext.get().get_last_update_interval()
    cur_nimg = int(resume_kimg * 1000)
    cur_tick = -1
    tick_start_nimg = cur_nimg
    prev_lod = -1.0
    running_mb_counter = 0
    while cur_nimg < total_kimg * 1000:
        if dnnlib.RunContext.get().should_stop(): break

        # Choose training parameters and configure training ops.
        sched = training_schedule(cur_nimg=cur_nimg, training_set=training_set, **sched_args)
        assert sched.minibatch_size % (sched.minibatch_gpu * num_gpus) == 0
        training_set.configure(sched.minibatch_gpu * num_gpus, sched.lod)
        if reset_opt_for_new_lod:
            if np.floor(sched.lod) != np.floor(prev_lod) or np.ceil(sched.lod) != np.ceil(prev_lod):
                G_opt.reset_optimizer_state(); D_opt.reset_optimizer_state()
        prev_lod = sched.lod

        if cur_nimg == 0:
            #print(rounds)
            print(f"batch size={sched.minibatch_size}, batch_gpu={sched.minibatch_gpu}, n_gpu={num_gpus}")
        # Run training ops.
        feed_dict = {lod_in: sched.lod, lrate_in: sched.G_lrate}
        for _repeat in range(minibatch_repeats):
            rounds = range(0, sched.minibatch_size, sched.minibatch_gpu * num_gpus)
            run_G_reg = (lazy_regularization and running_mb_counter % G_reg_interval == 0)
            run_D_reg = (lazy_regularization and running_mb_counter % D_reg_interval == 0)
            cur_nimg += sched.minibatch_size
            running_mb_counter += 1
            # Slow path with gradient accumulation or fast path without gradient accumulation, depending on "rounds".
            for _round in rounds:
                reals, labels = training_set.get_minibatch_np()
                masks = training_set.get_random_masks_np()
                feed_dict = {lod_in: sched.lod, lrate_in: sched.G_lrate,
                             reals_batch: reals, labels_batch: labels, masks_batch: masks}
                if run_G_reg:
                    # tflib.run([data_fetch_op, G_train_op, G_reg_op], feed_dict)
                    _, _, _, l = tflib.run([data_fetch_op, G_train_op, G_reg_op, G_loss], feed_dict)
                    # print(f"G_loss: {l}")
                else:
                    # tflib.run([data_fetch_op, G_train_op], feed_dict)
                    _, _, l = tflib.run([data_fetch_op, G_train_op, G_loss], feed_dict)
            tflib.run(Gs_update_op, feed_dict)
            for _round in rounds:
                reals, labels = training_set.get_minibatch_np()
                masks = training_set.get_random_masks_np()
                feed_dict = {lod_in: sched.lod, lrate_in: sched.G_lrate,
                             reals_batch: reals, labels_batch: labels, masks_batch: masks}
                if run_D_reg:
                    # tflib.run([data_fetch_op, D_train_op, D_reg_op], feed_dict)
                    _, _, _, l, l_r = tflib.run([data_fetch_op, D_train_op, D_reg_op, D_loss, D_reg], feed_dict)
                    # print(f"D_loss: {l}. D_reg: {l_r}")
                else:
                    # tflib.run([data_fetch_op, D_train_op], feed_dict)
                    _, _, l = tflib.run([data_fetch_op, D_train_op, D_loss], feed_dict)

        # Perform maintenance tasks once per tick.
        done = (cur_nimg >= total_kimg * 1000)
        if cur_tick < 0 or cur_nimg >= tick_start_nimg + sched.tick_kimg * 1000 or done:
            cur_tick += 1
            tick_kimg = (cur_nimg - tick_start_nimg) / 1000.0
            tick_start_nimg = cur_nimg
            tick_time = dnnlib.RunContext.get().get_time_since_last_update()
            total_time = dnnlib.RunContext.get().get_time_since_start() + resume_time

            # Report progress.
            print('tick %-5d kimg %-8.1f lod %-5.2f minibatch %-4d time %-12s sec/tick %-7.1f sec/kimg %-7.2f maintenance %-6.1f gpumem %.1f' % (
                autosummary('Progress/tick', cur_tick),
                autosummary('Progress/kimg', cur_nimg / 1000.0),
                autosummary('Progress/lod', sched.lod),
                autosummary('Progress/minibatch', sched.minibatch_size),
                dnnlib.util.format_time(autosummary('Timing/total_sec', total_time)),
                autosummary('Timing/sec_per_tick', tick_time),
                autosummary('Timing/sec_per_kimg', tick_time / tick_kimg),
                autosummary('Timing/maintenance_sec', maintenance_time),
                autosummary('Resources/peak_gpu_mem_gb', peak_gpu_mem_op.eval() / 2**30)))
            autosummary('Timing/total_hours', total_time / (60.0 * 60.0))
            autosummary('Timing/total_days', total_time / (24.0 * 60.0 * 60.0))
            #wandb
            wandb.log({"tick": cur_tick,
                       "kimg": cur_nimg / 1000.0,
                       "lod": sched.lod,
                       "minibatch": sched.minibatch_size,
                       "time": total_time,
                       "sec/tick": tick_time,
                       "sec/kimg": tick_time / tick_kimg,
                       "maintenance": maintenance_time})

            # Save snapshots.
            if image_snapshot_ticks is not None and (cur_tick % image_snapshot_ticks == 0 or done):
                if args.save_image_grids:
                    grid_fakes = Gs.run(grid_latents, grid_labels, grid_reals, grid_masks, minibatch_size=sched.minibatch_gpu)
                    img_path = dnnlib.make_run_dir_path('fakes%06d.png' % (cur_nimg // 1000))
                    misc.save_image_grid(grid_fakes, img_path, drange=drange_net, grid_size=grid_size, pix2pix=training_set.pix2pix)
                    wandb.log({"fakes": wandb.Image(img_path, caption="Fakes")})
                log_images(args, model=Gs,
                           training_set=training_set,
                           viz_batch_processed=viz_batch_processed,
                           drange_net=drange_net)
            if network_snapshot_ticks is not None and (cur_tick % network_snapshot_ticks == 0 or done):
                #pkl = dnnlib.make_run_dir_path('network-snapshot-%06d.pkl' % (cur_nimg // 1000))
                model_path = os.path.join("checkpoints", wandb.run.id, f'network-snapshot-{cur_nimg // 1000:06d}.pkl')
                os.makedirs(os.path.dirname(model_path), exist_ok=True)
                misc.save_pkl((G, D, Gs), model_path)
                #metrics.run(pkl, run_dir=dnnlib.make_run_dir_path(), data_dir=dnnlib.convert_path(data_dir), num_gpus=num_gpus, tf_config=tf_config)

            # Update summaries and RunContext.
            metrics.update_autosummaries()
            tflib.autosummary.save_summaries(summary_log, cur_nimg)
            dnnlib.RunContext.get().update('%.2f' % sched.lod, cur_epoch=cur_nimg // 1000, max_epoch=total_kimg)
            maintenance_time = dnnlib.RunContext.get().get_last_update_interval() - tick_time

    # Save final snapshot.
    misc.save_pkl((G, D, Gs), dnnlib.make_run_dir_path('network-final.pkl'))

    # All done.
    summary_log.close()
    training_set.close()

#----------------------------------------------------------------------------
