import os
import sys
import cv2
import time
import argparse
import numpy as np
import tensorflow as tf

from net.vqvae import vq_encoder_spec, vq_decoder_spec
from net.structure_generator import structure_condition_spec, structure_pixelcnn_spec
from net.texture_generator import texture_generator_spec, texture_discriminator_spec
from data_loader import NewDataset
from vae_helpers.baseline_utils import update_args

# -----------------------------------------------------------------------------
parser = argparse.ArgumentParser()

# Data
parser.add_argument('--dataset', type=str, default='ffhq256', 
                    help='dataset of the experiment.')
parser.add_argument('--full_model_dir', type=str, default='saved_vaes/ffhq256/full/', 
                    help='full model is given here.')

# Architecture
parser.add_argument('--image_size', type=int, default=256,
                    help='provide square images of this size.')
parser.add_argument('--nr_channel_vq', type=int, default=128,
                    help='number of channels in VQVAE.')
parser.add_argument('--nr_res_block_vq', type=int, default=2,
                    help='number of residual blocks in VQVAE.') 
parser.add_argument('--nr_res_channel_vq', type=int, default=64,
                    help='number of channels in the residual block in VQVAE.')
parser.add_argument('--nr_channel_s', type=int, default=128, 
                    help='number of channels in structure pixelcnn.')
parser.add_argument('--nr_res_channel_s', type=int, default=128,
                    help='number of channels in the residual block in structure pixelcnn.')
parser.add_argument('--nr_resnet_s', type=int, default=20, 
                    help='number of residual blocks in structure pixelcnn.')
parser.add_argument('--nr_resnet_out_s', type=int, default=20, 
                    help='number of output residual blocks in structure pixelcnn.')
parser.add_argument('--nr_attention_s', type=int, default=4, 
                    help='number of attention blocks in structure pixelcnn.')
parser.add_argument('--nr_head_s', type=int, default=8, 
                    help='number of attention heads in attention blocks.')
parser.add_argument('--nr_channel_cond_s', type=int, default=32, 
                    help='number of channels in structure condition network.')
parser.add_argument('--nr_res_channel_cond_s', type=int, default=32, 
                    help='number of channels in the residual block of structure condition network.')
parser.add_argument('--resnet_nonlinearity', type=str, default='concat_elu', 
                    help='nonlinearity in structure generator. One of "concat_elu", "elu", "relu". ')
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
parser = update_args(parser)
parser.add_argument('--batch_size', type=int, default=2)
parser.add_argument('--num_workers', type=int, default=0,
                    help="Number of workers for mask_generator sampler. If not given, the main process will sample the masks. Recommended: 8")
args = parser.parse_args()

print('------------ Options -------------')
for k, v in sorted(vars(args).items()):
    print('%s: %s' % (str(k), str(v)))
print('-------------- End ----------------')

################### Build structure generator & texture generator ###################
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

# Create structure generator
structure_condition = tf.make_template('structure_condition', structure_condition_spec)
structure_condition_opt = {'nr_channel': args.nr_channel_cond_s, 
                           'nr_res_channel': args.nr_res_channel_cond_s, 
                           'resnet_nonlinearity': args.resnet_nonlinearity}

structure_pixelcnn = tf.make_template('structure_pixelcnn', structure_pixelcnn_spec)
structure_pixelcnn_opt = {'nr_channel': args.nr_channel_s,
                          'nr_res_channel': args.nr_res_channel_s,
                          'nr_resnet': args.nr_resnet_s,
                          'nr_out_resnet': args.nr_resnet_out_s,
                          'nr_attention': args.nr_attention_s,
                          'nr_head': args.nr_head_s,
                          'resnet_nonlinearity': args.resnet_nonlinearity,
                          'num_embeddings': args.num_embeddings}

# Create texture generator
texture_generator = tf.make_template('texture_generator', texture_generator_spec)
texture_generator_opt = {'nr_channel': args.nr_channel_gen_t}

texture_discriminator = tf.make_template('texture_discriminator', texture_discriminator_spec)
texture_discriminator_opt = {'nr_channel': args.nr_channel_dis_t}

# Sample structure feature maps
top_shape = (args.image_size//8, args.image_size//8, 1)
img_ph = tf.placeholder(tf.float32, shape=(1, args.image_size, args.image_size, 3))
mask_ph = tf.placeholder(tf.float32, shape=(1, args.image_size, args.image_size, 1))
e_sample = tf.placeholder(tf.float32, shape=(1, args.image_size//8, args.image_size//8, args.embedding_dim))
h_sample = tf.placeholder(tf.float32, shape=(1, args.image_size//8, args.image_size//8, 8*args.nr_channel_cond_s))

batch_pos = img_ph
mask = mask_ph
masked = batch_pos * (1. - mask)
enc_gt = vq_encoder(batch_pos, is_training=False, **vq_encoder_opt)
dec_gt = vq_decoder(enc_gt['quant_t'], enc_gt['quant_b'], **vq_decoder_opt)
cond_masked = structure_condition(masked, mask, **structure_condition_opt)
pix_out = structure_pixelcnn(e_sample, h_sample, dropout_p=0., **structure_pixelcnn_opt)
pix_out = tf.reshape(pix_out, (-1, args.num_embeddings))
probs_out = tf.nn.log_softmax(pix_out, axis=-1)
samples_out = tf.multinomial(probs_out, 1)
samples_out = tf.reshape(samples_out, (-1, ) + top_shape[:-1])
new_e_gen = tf.nn.embedding_lookup(tf.transpose(enc_gt['embed_t'], [1, 0]), samples_out, validate_indices=False)

# Inpaint with generated structure feature maps
gen_out = texture_generator(masked, mask, e_sample, **texture_generator_opt)
img_gen = gen_out * mask + masked * (1. - mask)

# Discriminator
dis_out = texture_discriminator(tf.concat([img_gen, mask], axis=3), **texture_discriminator_opt)

# sample from the model
def sample_from_model(sess, img_np, mask_np):
    cond_masked_np = sess.run(cond_masked, {img_ph: img_np, mask_ph: mask_np})
    feed_dict = {h_sample: cond_masked_np}
    e_gen = np.zeros((1, args.image_size//8, args.image_size//8, args.embedding_dim), dtype=np.float32)
    for yi in range(top_shape[0]):
        for xi in range(top_shape[1]):
            feed_dict.update({e_sample: e_gen})
            new_e_gen_np = sess.run(new_e_gen, feed_dict)
            e_gen[:,yi,xi,:] = new_e_gen_np[:,yi,xi,:]
    img_gen_np = sess.run(img_gen, {img_ph: img_np, mask_ph: mask_np, e_sample: e_gen})
    return img_gen_np

################### Evaluate test images ###################
# Create a saver to restore full model
restore_saver = tf.train.Saver()

# TF session
sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
# Restore full model
ckpt = tf.train.get_checkpoint_state(args.full_model_dir)
if ckpt and ckpt.model_checkpoint_path:
    restore_saver.restore(sess, ckpt.model_checkpoint_path)
    print('Full model restored ...')
else:
    print('Restore full model failed! EXIT!')
    sys.exit()
 
dataset = NewDataset(args)
inpainted_list = []
import pdb; pdb.set_trace()
for i in range(10):
    t_im = dataset.get_minibatch_np()
    msk = dataset.get_random_masks_np(minibatch_size=1)
    img_gen_np = sample_from_model(sess, t_im[:1], msk)
    output = ((img_gen_np[0] + 1.) * 127.5).astype(np.uint8)
    inpainted_list.append(output)

for i in range(10):
    begin = time.time()
    img_name = img_list[i]
    mask_name = mask_list[i]
    img_np = cv2.imread(img_name)[:,:,::-1].astype(np.float)
    img_np = cv2.resize(img_np, (args.image_size, args.image_size), interpolation=cv2.INTER_LINEAR)
    mask_np = cv2.imread(mask_name, cv2.IMREAD_GRAYSCALE).astype(np.float)
    mask_np = np.expand_dims(mask_np, -1)

    # Normalize and reshape the image and mask
    img_np = img_np / 127.5 - 1.
    mask_np = mask_np / 255.
    img_np = np.expand_dims(img_np, 0)
    mask_np = np.expand_dims(mask_np, 0)

    # Run the result
    img_gen_np = sample_from_model(sess, img_np, mask_np)
    output = ((img_gen_np[0] + 1.) * 127.5).astype(np.uint8)

    # Save inpainting results into save directory
    cv2.imwrite(os.path.join(folder_path, '%05d.png' % i), output[:,:,::-1])
    print('%05d.png is generated. time: %.2fs.' % (i, time.time() - begin))
    sys.stdout.flush()
