import os
import sys
import time
import argparse
import numpy as np
import tensorflow as tf

from data_loader import NewDataset
from net.vqvae import vq_encoder_spec, vq_decoder_spec
from net.texture_generator import texture_generator_spec, texture_discriminator_spec
import net.nn as nn

from vae_helpers.baseline_utils import update_args
from vae_helpers import rng_decorator
import wandb


PROJECT_NAME = 'vq-vae'
if "--unobserve" in sys.argv:
    sys.argv.remove("--unobserve")
    os.environ["WANDB_MODE"] = "dryrun"


@rng_decorator(0)
def log_images(sess, dataset):
    v_im = dataset.get_minibatch_val_np()
    msk = dataset.get_random_masks_np(minibatch_size=1)
    # Inpaint images
    gt_np, masked_np, complete_np, recons_gt_np = sess.run([valid_images, batch_incomplete, batch_complete, recons_gt],
                                                           {valid_images: v_im, mask:msk})

    """ Show 4 inpainting images under guidance of ground truth structure feature maps in the training of texture generator."""
    log_dict = {}
    for idx in range(4):
        gt_i = ((gt_np[idx] + 1.) * 127.5).astype(np.uint8)
        masked_i = ((masked_np[idx] + 1.) * 127.5).astype(np.uint8)
        complete_i = ((complete_np[idx] + 1.) * 127.5).astype(np.uint8)
        recons_gt_i = ((recons_gt_np[idx] + 1.) * 127.5).astype(np.uint8)
        report_img = np.concatenate([gt_i, masked_i, complete_i, recons_gt_i], axis=1)
        caption = f"Sample {idx}"
        log_dict.update({
            caption: wandb.Image(report_img, caption=caption)})
    wandb.log(log_dict)


# -----------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Data
    parser.add_argument('--checkpoints_dir', type=str, default='checkpoints',
                        help='checkpoints are saved here.')
    parser.add_argument('--dataset', type=str, required='True',
                            choices=['ffhq256', 'cifar10'],
                            help='dataset of the experiment.')
    parser.add_argument('--vqvae_network_dir', type=str, default=None, required=True,
                        help='pre-trained VQVAE network is given here.')

    # Architecture
    parser.add_argument('--load_size', type=int, default=266,
                        help='scale images to this size.')
    parser.add_argument('--image_size', type=int, default=256,
                        help='provide square images of this size.')
    parser.add_argument('--nr_channel_vq', type=int, default=128,
                        help='number of channels in VQVAE.')
    parser.add_argument('--nr_res_block_vq', type=int, default=2,
                        help='number of residual blocks in VQVAE.') 
    parser.add_argument('--nr_res_channel_vq', type=int, default=64,
                        help='number of channels in the residual block in VQVAE.')
    parser.add_argument('--nr_channel_gen_t', type=int, default=64, 
                        help='number of channels in texture generator.')
    parser.add_argument('--nr_channel_dis_t', type=int, default=64, 
                        help='number of channels in texture discriminator.')

    # Vector quantizer
    parser.add_argument('--embedding_dim', type=int, default=64, 
                        help='number of the dimensions of embeddings in vector quantizer.')
    parser.add_argument('--num_embeddings', type=int, default=512, 
                        help='number of embeddings in vector quantizer.')
    parser.add_argument('--commitment_cost', type=float, default=0.25,
                        help='weight of commitment loss in vector quantizer.')
    parser.add_argument('--decay', type=float, default=0.99,
                        help='decay of EMA updates in vector quantizer.')

    # Training setting
    parser.add_argument('--batch_size', type=int, default=8, 
                        help='batch size.')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                        help='learning rate.')
    parser.add_argument('--fea_loss_weight', type=float, default=0.1, 
                        help='weight of feature loss.')
    parser.add_argument('--max_steps', type=int, default=1000000,
                        help='max number of iterations.')
    parser.add_argument('--train_spe', type=int, default=50000,
                        help='steps of inpainting images and saving models.')

    # EMA setting
    parser.add_argument('--ema_decay', type=float, default=0.9997, 
                        help='decay rate of EMA in validation.')

    # Mask setting
    parser.add_argument('--random_mask', type=bool, default=False,
                        help='random mask or not.')
    parser.add_argument('--mask_size', type=int, default=128,
                        help='provide square masks of this size.')
    parser.add_argument('--max_delta', type=int, default=0,
                        help='max delta of masks.')
    parser.add_argument('--margins', type=int, default=0,
                        help='margins of masks.')
    parser = update_args(parser)

    # New arguments
    parser.add_argument('--num_workers', type=int, default=8,
                        help="Number of workers for mask_generator sampler. If not given, the main process will sample the masks. Recommended: 8")

    args = parser.parse_args()
    assert args.vqvae_network_dir is not None, "A required argument is missing: --vqvae_network_dir"
    assert os.path.exists(args.vqvae_network_dir), f"Trained VQVAE weights directory does not exist ({args.vqvae_network_dir})"

    wandb.init(project=PROJECT_NAME, entity=os.environ['WANDB_ENTITY'],
               config=args, tags=["train_texture_generator"] + args.tags,
               sync_tensorboard=True, settings=wandb.Settings(start_method='fork'))

    print('------------ Options -------------')
    for k, v in sorted(vars(args).items()):
        print('%s: %s' % (str(k), str(v)))
    print('-------------- End ----------------')

    # -----------------------------------------------------------------------------
    # Create save folder
    folder_path = os.path.join(args.checkpoints_dir, f"{wandb.run.id}")

    os.makedirs(folder_path, exist_ok=True)

    # Data loader
    # train_loader = DataLoader(flist=args.train_flist, 
    #                         batch_size=args.batch_size,
    #                         o_size=args.load_size,
    #                         im_size=args.image_size,
    #                         is_train=True)
    # valid_loader = DataLoader(flist=args.valid_flist, 
    #                         batch_size=args.batch_size,
    #                         o_size=args.load_size,
    #                         im_size=args.image_size,
    #                         is_train=False)
    train_images = tf.placeholder(tf.float32, name='train', shape=[args.batch_size, args.image_size, args.image_size, 3])
    valid_images = tf.placeholder(tf.float32, name='valid', shape=[args.batch_size, args.image_size, args.image_size, 3]) 
    mask = tf.placeholder(tf.float32, name='mask', shape=[1, args.image_size, args.image_size, 1])

    # train_images = train_loader.load_items()
    # valid_images = valid_loader.load_items()

    # train_iterator = train_loader.iterator
    # valid_iterator = valid_loader.iterator

    # Generate mask, 1 represents masked point
    # if args.random_mask:
    #     bbox = nn.random_bbox(args.image_size, args.image_size, args.margins, args.mask_size, random_mask=True)
    #     regular_mask = nn.bbox2mask(bbox, args.image_size, args.image_size, args.max_delta, name='mask_c')
    #     irregular_mask = nn.brush_stroke_mask(args.image_size, args.image_size, name='mask_c')
    #     mask = tf.cond(tf.less(tf.random_uniform((1,), dtype=tf.float32)[0], 0.5),
    #                             lambda: regular_mask,
    #                             lambda: irregular_mask)

    # else:
    #     bbox = nn.random_bbox(args.image_size, args.image_size, args.margins, args.mask_size, random_mask=False)
    #     regular_mask = nn.bbox2mask(bbox, args.image_size, args.image_size, args.max_delta, name='mask_c')
    #     mask = regular_mask

    ################### Build texture generator ###################
    # Create VQVAE network
    vq_encoder = tf.make_template('vq_encoder', vq_encoder_spec)
    vq_encoder_opt = {'nr_channel': args.nr_channel_vq, 
                      'nr_res_block': args.nr_res_block_vq,
                      'nr_res_channel': args.nr_res_channel_vq,
                      'embedding_dim': args.embedding_dim,
                      'num_embeddings': args.num_embeddings,
                      'commitment_cost': args.commitment_cost,
                      'decay': args.decay}

    vq_decoder = tf.make_template('vq_decoder', vq_decoder_spec)
    vq_decoder_opt = {'nr_channel': args.nr_channel_vq, 
                      'nr_res_block': args.nr_res_block_vq,
                      'nr_res_channel': args.nr_res_channel_vq,
                      'embedding_dim': args.embedding_dim}

    # Create texture generator & discriminator
    texture_generator = tf.make_template('texture_generator', texture_generator_spec)
    texture_generator_opt = {'nr_channel': args.nr_channel_gen_t}

    texture_discriminator = tf.make_template('texture_discriminator', texture_discriminator_spec)
    texture_discriminator_opt = {'nr_channel': args.nr_channel_dis_t}

    # Variables to restore
    batch_pos = train_images
    enc_gt = vq_encoder(batch_pos, is_training=False, **vq_encoder_opt)
    dec_gt = vq_decoder(enc_gt['quant_t'], enc_gt['quant_b'], **vq_decoder_opt)
    autoencoder_params = []
    for v in tf.trainable_variables():
        if 'vector_quantize' not in v.name:
            autoencoder_params.append(v)
    ema = tf.train.ExponentialMovingAverage(decay=args.ema_decay)
    variables_to_restore = ema.variables_to_restore(moving_avg_variables=autoencoder_params)

    # Train
    # Texture generator
    # We use structure feature maps of ground truth when training texture generator
    batch_pos = train_images
    masked = batch_pos * (1. - mask) 
    enc_gt = vq_encoder(batch_pos, is_training=False, **vq_encoder_opt)
    gen_out = texture_generator(masked, mask, enc_gt['quant_t'], **texture_generator_opt)
    ae_loss = tf.reduce_mean(tf.abs(batch_pos - gen_out))
    # Texture discriminator
    batch_complete = gen_out * mask + masked * (1. - mask)
    pos_neg = tf.concat([batch_pos, batch_complete], axis=0)
    pos_neg = tf.concat([pos_neg, tf.tile(mask, [args.batch_size*2, 1, 1, 1])], axis=3)
    dis_out = texture_discriminator(pos_neg, **texture_discriminator_opt)
    dis_pos, dis_neg = tf.split(dis_out, 2)
    # Hinge loss
    hinge_pos = tf.reduce_mean(tf.nn.relu(1-dis_pos))
    hinge_neg = tf.reduce_mean(tf.nn.relu(1+dis_neg))
    dis_loss = tf.add(.5 * hinge_pos, .5 * hinge_neg)
    gen_loss = - tf.reduce_mean(dis_neg)
    # Feature loss
    enc_complete = vq_encoder(batch_complete, is_training=False, **vq_encoder_opt)
    fea_loss1 = args.fea_loss_weight * nn.feature_loss(enc_complete['enc_t'], enc_gt['idx_t'], enc_gt['embed_t'])
    fea_loss2 = args.fea_loss_weight * nn.feature_loss(enc_complete['enc_b'], enc_gt['idx_b'], enc_gt['embed_b'])
    fea_loss = fea_loss1 + fea_loss2
    gen_loss = gen_loss + ae_loss + fea_loss

    # Variables of texture generator & discriminator
    texture_generator_params = []
    texture_discriminator_params = []
    for v in tf.trainable_variables():
        if 'texture_generator' in v.name:
            texture_generator_params.append(v)
        elif 'texture_discriminator' in v.name:
            texture_discriminator_params.append(v)  

    # Create optimizer
    tf_lr = tf.placeholder(tf.float32, shape=[])
    gen_optimizer = tf.train.AdamOptimizer(learning_rate=tf_lr, beta1=0.5, beta2=0.999)
    dis_optimizer = gen_optimizer
    gen_op = gen_optimizer.minimize(gen_loss, var_list=texture_generator_params)
    dis_op = dis_optimizer.minimize(dis_loss, var_list=texture_discriminator_params)

    # Valid
    # Texture generator
    batch_pos = valid_images
    masked = batch_pos * (1. - mask) 
    batch_incomplete = masked
    enc_gt = vq_encoder(batch_pos, is_training=False, **vq_encoder_opt)
    gen_out = texture_generator(masked, mask, enc_gt['quant_t'], **texture_generator_opt)
    batch_complete = gen_out * mask + masked * (1. - mask)
    # Visualization of ground truth structure feature maps
    dec_gt = vq_decoder(enc_gt['quant_t'], tf.zeros_like(enc_gt['quant_b'], dtype=tf.float32), **vq_decoder_opt)
    recons_gt = tf.clip_by_value(dec_gt['dec_b'], -1, 1)

    ################### Train texture generator ###################
    # Create a saver to restore VQVAE network
    restore_saver = tf.train.Saver(variables_to_restore)

    # Create a saver to save VQVAE & texture generator
    saver = tf.train.Saver(tf.global_variables(), max_to_keep=None)

    # TF session
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        # Initialize dataset
        dataset = NewDataset(args)
        
        # Initialize global variables
        sess.run(tf.global_variables_initializer())

        # Restore pre-trained VQVAE network
        ckpt = tf.train.get_checkpoint_state(args.vqvae_network_dir)
        if ckpt and ckpt.model_checkpoint_path:
            restore_saver.restore(sess, ckpt.model_checkpoint_path)
            print('VQVAE network restored ...')
        else:
            print('Restore VQVAE network failed! EXIT!')
            sys.exit()

        # Start to train
        gen_losses = []
        lr = args.learning_rate
        begin = time.time()
        for i in range(args.max_steps):
            t_im = dataset.get_minibatch_np()
            msk = dataset.get_random_masks_np(minibatch_size=1)
            # Generator update
            result = sess.run([gen_op, gen_loss], {tf_lr:lr,
                                                   train_images: t_im,
                                                   mask: msk})
            # Discriminator update
            t_im = dataset.get_minibatch_np()
            msk = dataset.get_random_masks_np(minibatch_size=1)
            sess.run(dis_op, {tf_lr:lr, train_images: t_im, mask: msk})
            gen_losses.append(result[1])
            
            # Print training loss every 100 iterations
            if (i + 1) % 100 == 0:
                l_g = np.mean(gen_losses[-100:])
                tm = time.time() - begin
                print('%d iterations, time: %ds, gen loss: %.5f.' % 
                    ((i + 1), tm, l_g))
                sys.stdout.flush()
                wandb.log({"loss/rec": l_g,
                           "time/iter": tm,
                           "step": i+1})
                begin = time.time()

            # Inpaint images & Save model
            if (i + 1) % args.train_spe == 0:
                # Inpaint images
                log_images(sess, dataset)
                tm = time.time() - begin
                # Print inpainting time
                print('%d iterations, inpainting time: %.3fs' % ((i + 1), tm))
                sys.stdout.flush()
                wandb.log({"time/recon": tm,
                           "step": i+1})

                # Save model
                checkpoint_path = os.path.join(folder_path, f'model_{i+1}.ckpt')
                saver.save(sess, checkpoint_path)
                begin = time.time()
