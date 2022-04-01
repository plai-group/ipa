import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import torch
from torch.distributions import Normal
import os, sys
from tqdm import tqdm
from argparse import ArgumentParser
import wandb

from ablation.helper import list2mask, get_results_dir
from neural_process import NeuralProcessImg
import datasets
import utils
from utils import to_rgb, inpaint_recreate
from torchvision.utils import make_grid
from torch import nn
# Inception score and FID imports
from misc.inception import inception_score, inception_score_mnist
from misc.pytorch_fid import calculate_fid_no_paths
from misc import pytorch_fid
from torchvision.models.inception import inception_v3, Inception3
import json
from test_utils import load_trained_model_and_config
from torchvision.models.resnet import ResNet, BasicBlock, Bottleneck


N_LIST = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 30, 40, 50, 100, 150, 200, 300, 400, 500, 1024]
TEST_IDX_LIST = [22, 28, 303, 82, 0, 1, 2]#, 15, 9, 30, 84]
MODE_NAMES = {"is": "is",
              "is_mnist": "ismnist",
              "is_celeba": "isceleba",
              "is_celeba_new": "iscelebanew",
              "is_fashion_mnist": "isfashionmnist"}
RANDOM_STATE = 123


class MNISTResNet(ResNet):
    def __init__(self):
        super(MNISTResNet, self).__init__(BasicBlock, [2, 2, 2, 2], num_classes=10) # Based on ResNet18
        # super(MNISTResNet, self).__init__(BasicBlock, [3, 4, 6, 3], num_classes=10) # Based on ResNet34
        # super(MNISTResNet, self).__init__(Bottleneck, [3, 4, 6, 3], num_classes=10) # Based on ResNet50
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=1, padding=3,bias=False)


def resnet34(num_classes): # CelebA classifier
    """Constructs a ResNet-34 model."""
    model = ResNet(block=BasicBlock, 
                   layers=[3, 4, 6, 3],
                   num_classes=num_classes)
    return model


class Generator:
    def __init__(self, sampler):
        self.sampler = sampler
    
    def sample(self, shape):
        return self.sampler(shape)


def get_generator(model, img, mask):
    """
    img (torch.Tensor): with shape CxWxH
    
    mask (torch.Tensor): with shape CxWxH
    
    Returns
    -------
    A Generator object that, when its sample function is called, generates
    conditional images
    """
    _, width, height = img.shape
    # Create the list of context masks
    with torch.no_grad():
        target_mask = torch.ones_like(mask) # All the pixels
        # Add a batch dimension to tensors and move to GPU
        img_batch = img.unsqueeze(0).to(device)
        context_batch = mask.unsqueeze(0).to(device)
        target_batch = target_mask.unsqueeze(0).to(device)
        x_target, _ = utils.img_mask_to_np_input(img_batch, target_batch)
        def sampler(shape):
            if isinstance(shape, int):
                shape = torch.Size([shape])
            n = shape.numel()
            with torch.no_grad():
                py = model(img_batch.expand(n, *img_batch.shape[1:]),
                           context_batch.expand(n, *context_batch.shape[1:]),
                           target_batch.expand(n, *target_batch.shape[1:]))
                img_rec_flat = utils.xy_to_img(x_target.expand(n, *x_target.shape[1:]),
                                               py.mean, img.size())
                img_rec_flat = torch.clamp(to_rgb(to_rgb(img_rec_flat)), 0, 1)
                res = img_rec_flat.view(*shape, *img_rec_flat.shape[1:])
                return res
    return Generator(sampler)


if __name__ == "__main__":
    os.environ["WANDB_MODE"] = "dryrun"
    wandb_run = wandb.init(project = "dummy")

    is_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if is_cuda else "cpu")
    RESULTS_WANDB_PROJECT_NAME = "STAIR-neural-process"

    parser = ArgumentParser()
    parser.add_argument('--run_name', default=None)
    parser.add_argument('--run_id', default=None)
    parser.add_argument('--dir', default="results")
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--mode', type=str, choices=["is", "is_mnist", "is_celeba", "is_celeba_new", "is_fashion_mnist"], required=True)
    parser.add_argument('--idx', default=None)
    parser.add_argument('--expectation', action="store_true")
    parser.add_argument('--entropy', action="store_true")
    args = parser.parse_args()
    if args.idx is not None:
        args.idx = int(args.idx)
    print(args.__dict__)
    assert (args.dataset is not  None) + (args.run_name is not None) + (args.run_id is not None) == 1 # Exactly one of dataset, run_name and run_id should be given
    default_dir = args.dir

    ## Load the runs
    api = wandb.Api()
    if args.run_name is None and args.run_id is None:
        run_list = api.runs(f"saeidnp/{RESULTS_WANDB_PROJECT_NAME}", {"$and": [{"config.dataset": args.dataset}, {"tags": {"$in": ["ANP", "det", "nodet", "cross", "self", "nodet_att"]}}]})
        run_list = [run for run in run_list if "v2" in run.tags]
        print(f"Found {len(run_list)} jobs")
    elif args.run_name is not None:
        run_list = api.runs(f"saeidnp/{RESULTS_WANDB_PROJECT_NAME}", {"displayName": args.run_name})
        assert len(run_list) == 1
    else:
        run_list = [api.run(f"saeidnp/{RESULTS_WANDB_PROJECT_NAME}/{args.run_id}")]

    # Load inception models
    if args.mode == "is":
        # Load pretrained inception model
        print("Loading pretrained inception model (for Inception score)")
        inception_model = inception_v3(pretrained=True,
                                       transform_input=False).to(device)
    elif args.mode == "is_mnist":
        mnist_classifier = MNISTResNet()
        mnist_classifier.load_state_dict(torch.load("ablation/mnist_classifier.pt", map_location=lambda storage, loc: storage))
        mnist_classifier.to(device)
        mnist_classifier.eval()
    elif args.mode == "is_celeba_new":
        mnist_classifier = resnet34(16) # It's actually celeba_classifier, but deadline is in less than 7 hours! so, who cares?
        mnist_classifier.load_state_dict(torch.load("ablation/celeba_classifier_new.pt", map_location=lambda storage, loc: storage))
        mnist_classifier.to(device)
        mnist_classifier.eval()
    elif args.mode == "is_fashion_mnist":
        mnist_classifier = MNISTResNet()
        mnist_classifier.load_state_dict(torch.load("ablation/fashion_mnist_classifier.pt", map_location=lambda storage, loc: storage))
        mnist_classifier.to(device)
        mnist_classifier.eval()
    else:
        assert args.mode == "is_celeba"
        inception_model = Inception3(transform_input=False)
        inception_model.fc = nn.Linear(2048, 40)
        state_dict = torch.load("ablation/inception_celeba.pt")
        inception_model.load_state_dict(state_dict)
        inception_model.to(device)
        inception_model.eval()

    for run in run_list:
        if run.state != "finished":
            continue
        if args.mode == "is_mnist" and run.config["dataset"] != "mnist":
            print(f"is_mnist is only applicable to runs with MNIST dataset. Skipping {run.name}...")
            continue
        print(f"Processing {run.name}")
        if "(" not in run.name:
            print("run name is not updated. Skipping...")
            continue
        args.dir = default_dir
        args.run_name = run.name
        NAME = args.run_name.split('(')[0].strip()
        # Load the NP model
        model, config, _ = load_trained_model_and_config(run, device, RESULTS_WANDB_PROJECT_NAME)
        if config["dataset"] == "mnist":
            assert args.mode == "is" or args.mode == "is_mnist"
        elif config["dataset"] == "celeba":
            assert args.mode == "is" or args.mode == "is_celeba" or args.mode == "is_celeba_new"
        elif config["dataset"] == "fashion_mnist":
            assert args.mode == "is" or args.mode == "is_fashion_mnist"
        else:
            raise Exception(f"Unexpected model dataset {config['dataset']}")

        # Prepaer path to output directory
        args.dir = get_results_dir(args.dir, run)
        os.makedirs(args.dir, exist_ok=True)
        print(f"Output directory: {args.dir}")
        ## Initialize the dataset
        test_set = datasets.DATASET_DICT[config["dataset"]](size=config["img_size"][1], split="test")

        test_idx_list = TEST_IDX_LIST if args.idx is None else [TEST_IDX_LIST[args.idx]]
        # Do the job!
        for idx in test_idx_list:
            if args.expectation:
                if args.entropy:
                    json_path = os.path.join(args.dir, f"newexpectation_{MODE_NAMES[args.mode]}_entropy_{NAME}_{idx}.json")
                else:
                    json_path = os.path.join(args.dir, f"newexpectation_{MODE_NAMES[args.mode]}_{NAME}_{idx}.json")
            else:
                if args.entropy:
                    json_path = os.path.join(args.dir, f"{MODE_NAMES[args.mode]}_entropy_{NAME}_{idx}.json")
                else:
                    json_path = os.path.join(args.dir, f"{MODE_NAMES[args.mode]}_{NAME}_{idx}.json")
            plot_path = f"{os.path.splitext(json_path)[0]}.pdf"
            # Skip if the plot already exists
            if os.path.exists(json_path):
                print(f"{json_path} already exists. Skipping ...")
                continue
            # Take the test image out of the dataset
            img = test_set[idx][0]
            x_vals, y_vals_mean, y_vals_std = [], [], []
            if args.expectation:
                rng = np.random.RandomState(RANDOM_STATE)
                for n in N_LIST:
                    x_vals.append(n)
                    print(f"@{n}")
                    y_vals_mean_tmp, y_vals_std_tmp = [], []
                    for _ in range(10):
                        mask = list2mask(rng.choice(range(1024), n, replace=False), img)
                        gen = get_generator(model, img, mask)
                        if args.mode == "is"  or args.mode == "is_celeba":
                            is_mean, is_std = inception_score(gen, N=int(5e4), cuda=is_cuda,
                                                              batch_size=32, resize=True, splits=10,
                                                              inception_model=inception_model,
                                                              entropy_only=args.entropy,
                                                              num_classes=40 if args.mode == "is_celeba" else 1000)
                        else:
                            is_mean, is_std = inception_score_mnist(gen, N=int(5e4), cuda=is_cuda,
                                                                    batch_size=32, splits=10,
                                                                    mnist_classifier=mnist_classifier,
                                                                    entropy_only=args.entropy,
                                                                    grayscale=(args.mode != "is_celeba_new"),
                                                                    num_classes=16 if args.mode=="is_celeba_new" else 10)
                        y_vals_mean_tmp.append(is_mean)
                        y_vals_std_tmp.append(is_std)
                    y_vals_mean.append(y_vals_mean_tmp)
                    y_vals_std.append(y_vals_std_tmp)
            else:
                # Prepare the list of context masks
                rng = np.random.RandomState(RANDOM_STATE)
                pixel_permutation = rng.permutation(range(32*32))
                mask_list = [list2mask([pixel_permutation[:i]], img) for i in N_LIST]
                for mask in mask_list:
                    x = mask.sum().item()
                    print(f"@{x}")
                    x_vals.append(x)
                    gen = get_generator(model, img, mask)
                    if args.mode == "is" or args.mode == "is_celeba":
                        is_mean, is_std = inception_score(gen, N=int(5e4), cuda=is_cuda,
                                                          batch_size=32, resize=True, splits=10,
                                                          inception_model=inception_model,
                                                          entropy_only=args.entropy,
                                                          num_classes=40 if args.mode == "is_celeba" else 1000)
                    else:
                        is_mean, is_std = inception_score_mnist(gen, N=int(5e4), cuda=is_cuda,
                                                                batch_size=32, splits=10,
                                                                mnist_classifier=mnist_classifier,
                                                                entropy_only=args.entropy,
                                                                grayscale=(args.mode == "is_mnist"),
                                                                num_classes=10 if args.mode=="is_mnist" else 16)
                    y_vals_mean.append(is_mean)
                    y_vals_std.append(is_std)
            data = dict(zip(x_vals, zip(y_vals_mean, y_vals_std)))
            # Save results JSON file
            print(f"Saving to {json_path}")
            with open(json_path, 'w') as outfile:
                json.dump(data, outfile)
            # Plot
            if args.expectation:
                mean = [np.mean(x) for x in y_vals_mean]
                std = [np.std(x) for x in y_vals_mean]
                mean = np.array(mean)
                std = np.array(std)
                plt.plot(x_vals, mean)
                plt.fill_between(x_vals, mean - std, mean + std, alpha=0.1)
                plt.xticks(x_vals)
                plt.xlabel("Context set size")
                plt.ylabel("Inception score")
                plt.savefig(plot_path)
                plt.close("all")
            else:
                y_vals_mean = np.array(y_vals_mean)
                y_vals_std = np.array(y_vals_std)
                plt.plot(x_vals, y_vals_mean)
                plt.fill_between(x_vals, y_vals_mean - y_vals_std, y_vals_mean + y_vals_std, alpha=0.1)
                plt.xticks(x_vals)
                plt.xlabel("Context set size")
                plt.ylabel("Inception score")
                plt.savefig(plot_path)
                plt.close("all")
