import os


def update_args(parser):
    parser.add_argument('--tags', type=str, nargs='*', default=[])
    parser.add_argument('--conditioning', type=str,
                        choices=['patches', 'patches-missing', 'blank', 'freeform'],
                        default='patches')
    parser.add_argument('--max_patches', type=int, default=5)
    parser.add_argument('--freeform_hole_range', type=float, nargs=2, default=[0, 1])
    parser.add_argument('--data_root', type=str, default='../../' if "DATA_ROOT" not in os.environ else os.environ["DATA_ROOT"])
    parser.add_argument('--patch_size_frac', type=float, default=0.35,
                        help="Patch width as fraction of image width.")
    parser.add_argument('--num_images_visualize', type=int, default=5)
    parser.add_argument('--num_samples_visualize', type=int, default=5)
    parser.add_argument('--mask_sample_workers', type=int, help='Parallelise mask sampling.')
    return parser