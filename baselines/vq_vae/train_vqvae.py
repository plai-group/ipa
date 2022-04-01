import os
import sys
import time
import argparse
import numpy as np
import tensorflow as tf

from data_loader import NewDataset
from net.vqvae import vq_encoder_spec, vq_decoder_spec
import net.nn as nn

from vae_helpers.baseline_utils import update_args
from vae_helpers import rng_decorator
import wandb


PROJECT_NAME = 'vq-vae'
if "--unobserve" in sys.argv:
    sys.argv.remove("--unobserve")
    os.environ["WANDB_MODE"] = "dryrun"


@rng_decorator(0)
def log_images(gt, recons):
    log_dict = {}
    for idx in range(5):
        gt_i = ((gt[idx] + 1.) * 127.5).astype(np.uint8)
        recons_i = ((recons[idx] + 1.) * 127.5).astype(np.uint8)
        caption = f"Sample {idx}"
        log_dict.update({caption: wandb.Image(np.concatenate([gt_i, recons_i], axis=1), caption=caption)})
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

    # Architecture
    parser.add_argument('--load_size', type=int, default=266,
                        help='scale images to this size.')
    parser.add_argument('--image_size', type=int, default=256,
                        help='then random crop to this size.')
    parser.add_argument('--nr_channel_vq', type=int, default=128,
                        help='number of channels in VQVAE.')
    parser.add_argument('--nr_res_block_vq', type=int, default=2,
                        help='number of residual blocks in VQVAE.') 
    parser.add_argument('--nr_res_channel_vq', type=int, default=64,
                        help='number of channels in the residual block in VQVAE.')
    
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
    parser.add_argument('--max_steps', type=int, default=1000000,
                        help='max number of iterations.')
    parser.add_argument('--val_steps', type=int, default=10000,
                        help='steps of validation.')
    parser.add_argument('--train_spe', type=int, default=50000,
                        help='steps of saving models.')

    # EMA setting
    parser.add_argument('--ema_decay', type=float, default=0.9997,
                        help='decay rate of EMA in validation.')

    # New arguments
    parser.add_argument('--num_workers', type=int, default=8,
                        help="Number of workers for mask_generator sampler. If not given, the main process will sample the masks. Recommended: 8")
    parser = update_args(parser)

    args = parser.parse_args()

    wandb.init(project=PROJECT_NAME, entity=os.environ['WANDB_ENTITY'],
            config=args, tags=["train_vqvae"] + args.tags, sync_tensorboard=True,
            settings=wandb.Settings(start_method='fork'))

    print('------------ Options -------------')
    for k, v in sorted(vars(args).items()):
        print('%s: %s' % (str(k), str(v)))
    print('-------------- End ----------------')

    # -----------------------------------------------------------------------------
    # Create save folder
    folder_path = os.path.join(args.checkpoints_dir, wandb.run.id)

    os.makedirs(folder_path, exist_ok=True)

    # Data loader
    # train_loader = DataLoader(flist=args.train_flist, 
    #                           batch_size=args.batch_size,
    #                           o_size=args.load_size,
    #                           im_size=args.image_size,
    #                           is_train=True)
    # valid_loader = DataLoader(flist=args.valid_flist, 
    #                           batch_size=args.batch_size,
    #                           o_size=args.load_size,
    #                           im_size=args.image_size,
    #                           is_train=False)
    train_images = tf.placeholder(tf.float32, name='train', shape=[args.batch_size, args.image_size, args.image_size, 3])
    valid_images = tf.placeholder(tf.float32, name='valid', shape=[args.batch_size, args.image_size, args.image_size, 3])

    ################### Build VQVAE network ###################
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

    # Train
    enc_out = vq_encoder(train_images, ema=None, is_training=True, **vq_encoder_opt)
    dec_out = vq_decoder(enc_out['quant_t'], enc_out['quant_b'], ema=None, **vq_decoder_opt)
    recons_loss = tf.reduce_mean((train_images - dec_out['dec_b'])**2)
    commit_loss = enc_out['loss']
    loss = recons_loss + commit_loss

    # Keep track of moving average
    autoencoder_params = []
    for v in tf.trainable_variables():
        if 'vector_quantize' not in v.name:
            autoencoder_params.append(v)
    ema = tf.train.ExponentialMovingAverage(decay=args.ema_decay)
    maintain_averages_op = tf.group(ema.apply(autoencoder_params))

    # Create optimizer
    tf_lr = tf.placeholder(tf.float32, shape=[])
    optimizer = tf.train.AdamOptimizer(learning_rate=tf_lr)
    train_op = tf.group(optimizer.minimize(loss), maintain_averages_op)

    # Valid
    enc_out = vq_encoder(valid_images, ema=ema, is_training=False, **vq_encoder_opt)
    dec_out = vq_decoder(enc_out['quant_t'], enc_out['quant_b'], ema=ema, **vq_decoder_opt)
    recons_loss_valid = tf.reduce_mean((valid_images - dec_out['dec_b'])**2)
    commit_loss_valid = enc_out['loss']
    recons_valid = tf.clip_by_value(dec_out['dec_b'], -1, 1)

    ################### Train VQVAE network ###################
    # Create a saver to save VQVAE network
    saver = tf.train.Saver(tf.global_variables(), max_to_keep=None)

    # TF session
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        # Initialize dataset
        dataset = NewDataset(args)
        
        # Initialize global variables
        sess.run(tf.global_variables_initializer())

        # Start to train
        train_recons_loss = []
        train_commit_loss = []
        lr = args.learning_rate
        begin = time.time()
        for i in range(args.max_steps):
            t_im = dataset.get_minibatch_np()
            result = sess.run([train_op, recons_loss, commit_loss],
                            {tf_lr:lr, train_images: t_im})
            train_recons_loss.append(result[1])
            train_commit_loss.append(result[2])
            
            # Print training loss every 100 iterations
            if (i + 1) % 100 == 0:
                l_r = np.mean(train_recons_loss[-100:])
                l_c = np.mean(train_commit_loss[-100:])
                tm = time.time() - begin
                print('%d iterations, time: %ds, ' % ((i + 1), tm) + 
                    'train_recons_loss: %.5f, train_commit_loss: %.5f.' %
                    (l_r, l_c))
                sys.stdout.flush()
                wandb.log({"loss/rec": l_r,
                           "loss/commit": l_c,
                           "time/iter": tm,
                           "step": i+1})
                begin = time.time()

            # Validate
            if (i + 1) % args.val_steps == 0:
                # Number of iterations every validation
                # Every iteration will evaluate (num_iter) batches of randomly cropped validation images
                num_iter = 100

                valid_recons_loss = []
                valid_commit_loss = []
                for step in range(num_iter):
                    v_im = dataset.get_minibatch_val_np()
                    valid_result = sess.run([recons_loss_valid, commit_loss_valid],
                                            {valid_images: v_im})
                    valid_recons_loss.append(valid_result[0])
                    valid_commit_loss.append(valid_result[1])

                # Print validation loss
                vl_r = np.mean(valid_recons_loss)
                vl_c = np.mean(valid_commit_loss)
                print('%d iterations, time: %ds, ' % ((i + 1), time.time()-begin) + 
                    'valid_recons_loss: %.5f, valid_commit_loss: %.5f.' %
                    (vl_r, vl_c))
                sys.stdout.flush()
                wandb.log({"validation/rec": vl_r,
                           "validation/commit": vl_c,
                           "step": i+1})
                begin = time.time()

            # Reconstruct images & Save model
            if (i + 1) % args.train_spe == 0:
                v_im = dataset.get_minibatch_val_np()
                # Reconstruct images
                recons_np = sess.run(recons_valid,
                                    {valid_images: v_im})
                log_images(v_im, recons_np)
                # Print reconstruction time
                tm = time.time() - begin
                print('%d iterations, reconstruction time: %.3fs.' % ((i + 1), tm))
                sys.stdout.flush()
                wandb.log({"time/recon": tm,
                           "step": i+1})

                # Save model
                checkpoint_path = os.path.join(folder_path, f'model_{i+1}.ckpt')
                saver.save(sess, checkpoint_path)
                begin = time.time()
