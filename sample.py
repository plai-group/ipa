import os
import torch
from PIL import Image
from train_helpers import set_up_hyperparams, load_vaes
from data import set_up_data


# load dataset and VAEs
H, logprint = set_up_hyperparams()
H.test_eval = True
H, data_train, data_valid_or_test, preprocess_fn = set_up_data(H)
_, ema_vae = load_vaes(H, logprint, ckpt_dir=H.ckpt_load_dir, ema_only=True)


# load and normalise test image
example_image = data_valid_or_test[0][0].unsqueeze(0)  # take first test image and add batch dimension
example_image, preprocessed = preprocess_fn([example_image])

# create observation mask
obs_mask = torch.zeros_like(preprocessed[:, :, :, :1])
B, H, W, C = preprocessed.shape
obs_mask[:, :H//2, :, :] = 1  # condition on top half
inp = torch.cat([preprocessed*obs_mask, obs_mask], dim=-1)

# run IPA
inp = inp.expand(5, -1, -1, -1)  # duplicate 5 times to get 5 samples
part_activations = ema_vae.part_encoder(inp)
sample_px_z, _ = ema_vae.decoder.run(
    sample_from='part',
    part_activations=part_activations
)
sample_batch = ema_vae.decoder.out_net.sample(sample_px_z)

# save samples
for i, sample in enumerate(sample_batch):
    samples_dir = './samples/'
    os.makedirs(samples_dir, exist_ok=True)
    Image.fromarray(sample).save(os.path.join(samples_dir, f"{i}.jpg"))
