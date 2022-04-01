import numpy as np

HPARAMS_REGISTRY = {}

def str2bool(s):
    return 't' in s.lower()
def int_or_none(s):
    return None if s.lower() == 'none' else int(s)


class Hyperparams(dict):
    def __getattr__(self, attr):
        try:
            return self[attr]
        except KeyError:
            print(f'Not a valid attribute {attr}. Returning None.')
            return None

    def __setattr__(self, attr, value):
        self[attr] = value

    def copy(self):
        new = Hyperparams()
        for k, v in self.items():
            new[k] = v
        return new


cifar10 = Hyperparams()
cifar10.width = 384
cifar10.lr = 0.0002
cifar10.zdim = 16
cifar10.wd = 0.01
cifar10.dec_blocks = "1x1,4m1,4x2,8m4,8x5,16m8,16x10,32m16,32x21"
cifar10.enc_blocks = "32x11,32d2,16x6,16d2,8x6,8d2,4x3,4d4,1x3"
cifar10.dataset = 'cifar10'
cifar10.n_batch = 16
cifar10.fid_bs = 16
cifar10.ema_rate = 0.9999
cifar10.viz_seed = 0
HPARAMS_REGISTRY['cifar10'] = cifar10

i32 = Hyperparams()
i32.update(cifar10)
i32.dataset = 'imagenet32'
i32.ema_rate = 0.999
i32.dec_blocks = "1x2,4m1,4x4,8m4,8x9,16m8,16x19,32m16,32x40"
i32.enc_blocks = "32x15,32d2,16x9,16d2,8x8,8d2,4x6,4d4,1x6"
i32.width = 512
i32.n_batch = 8
i32.fid_bs = 8
i32.lr = 0.00015
i32.grad_clip = 200.
i32.skip_threshold = 300.
i32.epochs_per_eval = 1
i32.epochs_per_eval_save = 1
HPARAMS_REGISTRY['imagenet32'] = i32

i64 = Hyperparams()
i64.update(i32)
i64.n_batch = 4
i64.fid_bs = 4
i64.grad_clip = 220.0
i64.skip_threshold = 380.0
i64.dataset = 'imagenet64'
i64.dec_blocks = "1x2,4m1,4x3,8m4,8x7,16m8,16x15,32m16,32x31,64m32,64x12"
i64.enc_blocks = "64x11,64d2,32x20,32d2,16x9,16d2,8x8,8d2,4x7,4d4,1x5"
HPARAMS_REGISTRY['imagenet64'] = i64

ffhq_256 = Hyperparams()
ffhq_256.update(i64)
ffhq_256.n_batch = 1
ffhq_256.fid_bs = 4
ffhq_256.lr = 0.00015
ffhq_256.dataset = 'ffhq_256'
ffhq_256.epochs_per_eval = 1
ffhq_256.epochs_per_eval_save = 1
ffhq_256.num_images_visualize = 3
ffhq_256.num_variables_visualize = 3
ffhq_256.num_temperatures_visualize = 1
ffhq_256.dec_blocks = "1x2,4m1,4x3,8m4,8x4,16m8,16x9,32m16,32x21,64m32,64x13,128m64,128x7,256m128"
ffhq_256.enc_blocks = "256x3,256d2,128x8,128d2,64x12,64d2,32x17,32d2,16x7,16d2,8x5,8d2,4x5,4d4,1x4"
ffhq_256.no_bias_above = 64
ffhq_256.grad_clip = 130.
ffhq_256.skip_threshold = 180.
ffhq_256.viz_seed = 1
HPARAMS_REGISTRY['ffhq256'] = ffhq_256

xray = Hyperparams()
xray.update(ffhq_256)
xray.dataset = 'xray'
xray.num_images_visualize = 5
HPARAMS_REGISTRY['xray'] = xray

shoes64 = Hyperparams()
shoes64.update(i64)
shoes64.dataset = 'shoes64'
shoes64.num_images_visualize = 5
HPARAMS_REGISTRY['shoes64'] = shoes64

bags64 = Hyperparams()
bags64.update(i64)
bags64.dataset = 'bags64'
bags64.num_images_visualize = 5
HPARAMS_REGISTRY['bags64'] = bags64

shoes = Hyperparams()
shoes.update(ffhq_256)
shoes.dataset = 'shoes'
shoes.num_images_visualize = 5
#shoes.dec_blocks = "1x2,4m1,4x3,8m4,8x4,16m8,16x9,32m16,32x21,64m32,64x1,128m64,128x1,256m128"
#shoes.enc_blocks = "256x1,256d2,128x1,128d2,64x1,64d2,32x17,32d2,16x7,16d2,8x5,8d2,4x5,4d4,1x4"
HPARAMS_REGISTRY['shoes'] = shoes

bags = Hyperparams()
bags.update(ffhq_256)
bags.dataset = 'bags'
bags.num_images_visualize = 5
#bags.dec_blocks = "1x2,4m1,4x3,8m4,8x4,16m8,16x9,32m16,32x21,64m32,64x1,128m64,128x1,256m128"
#bags.enc_blocks = "256x1,256d2,128x1,128d2,64x1,64d2,32x17,32d2,16x7,16d2,8x5,8d2,4x5,4d4,1x4"
HPARAMS_REGISTRY['bags'] = bags

ffhq1024 = Hyperparams()
ffhq1024.update(ffhq_256)
ffhq1024.dataset = 'ffhq_1024'
ffhq1024.data_root = './ffhq_images1024x1024'
ffhq1024.epochs_per_eval = 1
ffhq1024.epochs_per_eval_save = 1
ffhq1024.num_images_visualize = 1
ffhq1024.iters_per_images = 25000
ffhq1024.num_variables_visualize = 0
ffhq1024.num_temperatures_visualize = 4
ffhq1024.grad_clip = 360.
ffhq1024.skip_threshold = 500.
ffhq1024.num_mixtures = 2
ffhq1024.width = 16
ffhq1024.lr = 0.00007
ffhq1024.dec_blocks = "1x2,4m1,4x3,8m4,8x4,16m8,16x9,32m16,32x20,64m32,64x14,128m64,128x7,256m128,256x2,512m256,1024m512"
ffhq1024.enc_blocks = "1024x1,1024d2,512x3,512d2,256x5,256d2,128x7,128d2,64x10,64d2,32x14,32d2,16x7,16d2,8x5,8d2,4x5,4d4,1x4"
ffhq1024.custom_width_str = "512:32,256:64,128:512,64:512,32:512,16:512,8:512,4:512,1:512"
HPARAMS_REGISTRY['ffhq1024'] = ffhq1024

symobj = Hyperparams()
symobj.update(cifar10)
symobj.dataset = 'symobj'
HPARAMS_REGISTRY['symobj'] = symobj

symobj_simple = Hyperparams()
symobj_simple.update(symobj)
symobj_simple.dataset = 'symobj_simple'
HPARAMS_REGISTRY['symobj_simple'] = symobj_simple

ffhq32 = Hyperparams()
ffhq32.update(symobj)
ffhq32.dataset = 'ffhq32'
HPARAMS_REGISTRY['ffhq32'] = ffhq32

def parse_args_and_update_hparams(H, parser, s=None):
    args = parser.parse_args(s)
    valid_args = set(args.__dict__.keys())
    hparam_sets = [x for x in args.hparam_sets.split(',') if x]
    for hp_set in hparam_sets:
        hps = HPARAMS_REGISTRY[hp_set]
        for k in hps:
            if k not in valid_args:
                raise ValueError(f"{k} not in default args")
        parser.set_defaults(**hps)
    H.update(parser.parse_args(s).__dict__)
    H.conditional = (not H.unconditional)
    if H.seed is None:
        H.seed = np.random.randint(2**32-1)


def add_vae_arguments(parser):
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--port', type=int_or_none, default=29500)
    parser.add_argument('--save_dir', type=str, default='./saved_models')
    parser.add_argument('--data_root', type=str, default='./',
                        help="Will treat as an environment variable if it begins with $")
    parser.add_argument('--hparam_sets', '--hps', type=str)
    parser.add_argument('--desc', type=str, default='test')
    parser.add_argument('--ckpt_load_dir', type=str, default=None)
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--norm_like', type=str, default=None,
                        help='''Use normalisation stats from other dataset. Useful for transferring pretrained models.
                              Check data.py before using - this is implemented in an ad-hoc way.''')
    parser.add_argument('--ema_rate', type=float, default=0.999)
    parser.add_argument('--enc_blocks', type=str, default=None)
    parser.add_argument('--dec_blocks', type=str, default=None)
    parser.add_argument('--zdim', type=int, default=16)
    parser.add_argument('--width', type=int, default=512)
    parser.add_argument('--custom_width_str', type=str, default='')
    parser.add_argument('--bottleneck_multiple', type=float, default=0.25)
    parser.add_argument('--no_bias_above', type=int, default=64)
    parser.add_argument('--scale_encblock', action="store_true")
    parser.add_argument('--test_eval', action="store_true")
    parser.add_argument('--eval_with_train_set', action="store_true")
    parser.add_argument('--warmup_iters', type=float, default=100)
    parser.add_argument('--num_mixtures', type=int, default=10)
    parser.add_argument('--grad_clip', type=float, default=200.0)
    parser.add_argument('--skip_threshold', type=float, default=400.0)
    parser.add_argument('--lr', type=float, default=0.00015)
    parser.add_argument('--lr_prior', type=float, default=0.00015)
    parser.add_argument('--wd', type=float, default=0.0)
    parser.add_argument('--wd_prior', type=float, default=0.0)
    parser.add_argument('--num_epochs', type=int, default=10000)
    parser.add_argument('--num_iters', type=int, default=None,
                        help='Number of iterations to run for. Should only be used with num_epochs=1.')
    parser.add_argument('--n_batch', type=int, default=32)
    parser.add_argument('--fid_bs', type=int, default=None, help='Batch size to use on FID.')
    parser.add_argument('--n_workers', type=int, default=0)
    parser.add_argument('--grad_accumulations', type=int, default=1,
                        help='Set to >1 to simulate a bigger batch size.')
    parser.add_argument('--adam_beta1', type=float, default=0.9)
    parser.add_argument('--adam_beta2', type=float, default=0.9)
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--iters_per_backup', type=int, default=10000)
    parser.add_argument('--iters_per_ckpt', type=int, default=10000)
    parser.add_argument('--iters_per_save', type=int, default=10000)
    parser.add_argument('--iters_per_images', type=int, default=5000)
    parser.add_argument('--iters_per_log', type=int, default=100)
    parser.add_argument('--epochs_per_eval', type=int, default=10)
    parser.add_argument('--epochs_per_probe', type=int, default=None)
    parser.add_argument('--epochs_per_eval_save', type=int, default=20)
    parser.add_argument('--num_images_visualize', type=int, default=8)
    parser.add_argument('--num_variables_visualize', type=int, default=6)
    parser.add_argument('--num_temperatures_visualize', type=int, default=2)
    parser.add_argument('--num_reconstructions_visualize', type=int, default=3)
    parser.add_argument('--num_samples_visualize', type=int, default=5)
    parser.add_argument('--plot_ent', default=True, type=str2bool, help='Log plots of entropy of latent dists.')
    parser.add_argument('--wandb_id', type=str, default=None)
    # conditional training things
    parser.add_argument('--unconditional', action='store_true')
    parser.add_argument('--conditioning', type=str, default='freeform',
                        choices=['patches', 'patches-missing', 'blank', 'freeform', 'image'],
                        help='"image" should only be used with the Edges2Photos datasets.')
    parser.add_argument('--freeform_hole_range', type=float, nargs=2, default=[0, 1])
    parser.add_argument('--max_patches', type=int, default=5)
    parser.add_argument('--patch_size_frac', type=float, default=0.35,
                        help="Patch width as fraction of image width.")
    parser.add_argument('--pretrained_load_dir', type=str, default=None,
                        help='If provided, initializes from pretrained unconditional model.')
    parser.add_argument('--train_encoder_decoder', type=str, default="", choices=["", "slightly", "all"],
                        help='If True, will not fix weights of pretrained models.')
    parser.add_argument('--pretrained_partial_encoder', type=str, default="", choices=["", "mostly", "all"],
                        help='Initialise most/all weights in partial encoder to values from pretrained encoder.')
    parser.add_argument('--kl', type=str, default='r1_q', choices=['r1_q', 'q_r2'],
                        help='"r1_q" means training with the IPA objective, "q_r2" means training with IPA-R.')
    parser.add_argument('--rev_kl_schedule', type=int, nargs='*', default=None,
                        help='Iterations in which to introduce the reverse kl term. e.g. 10 20 will'+
                        'gradually introduce term between 10th and 20th iteration. ')
    parser.add_argument('--share_encoders', default=False, type=str2bool, help='Use partial encoder as full encoder.')
    parser.add_argument('--mask_distortion', action='store_true', help='Compute distortion for only unobserved pixels in forward KL loss.')
    parser.add_argument('--noisy_kl', action='store_true', help='Worse estimate of KL.')
    parser.add_argument('--clamp_std', default=None, type=float,
                        help='Clamp std for part encoder to potentially improve stability.')
    parser.add_argument('--likelihood_temp_schedule', type=float, nargs=3, default=[0, 1, 1],
                        help='n_iterations to anneal temperature to 1 over, and starting temperature. Only used for q_r2 KL divergence.')
    parser.add_argument('--tags', type=str, nargs='*', default=[], help='Tags for wandb run.')
    parser.add_argument('--no_ema', action="store_true",
                        help='If given, will ignore maintaining an EMA version of the network (will make the code faster; mainly used in sweeps)')
    parser.add_argument('--not_load_opt', action='store_true')
    parser.add_argument('--fid_samples', type=int_or_none, default=None, help='Number of samples to use for estimating FID score. If None, does not evaluate.')
    parser.add_argument('--viz_seed', type=int, default=0, help='Seed used to create masks for visualisation.')
    parser.add_argument('--mask_sample_workers', type=int_or_none, default=None, help='Parallelise mask sampling.')

    return parser
