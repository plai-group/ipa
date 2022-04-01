mkdir pretrained
mkdir pretrained/cifar10-1
wget https://openaipublic.blob.core.windows.net/very-deep-vaes-assets/vdvae-assets-2/cifar10-seed1-iter-1050000-model-ema.th
mv cifar10-seed1-iter-1050000-model-ema.th pretrained/cifar10-1/model-ema.th

mkdir pretrained/imagenet32
wget https://openaipublic.blob.core.windows.net/very-deep-vaes-assets/vdvae-assets/imagenet32-iter-1700000-model-ema.th
mv imagenet32-iter-1700000-model-ema.th pretrained/imagenet32/model-ema.th

mkdir pretrained/imagenet64
wget https://openaipublic.blob.core.windows.net/very-deep-vaes-assets/vdvae-assets-2/imagenet64-iter-1600000-model-ema.th
mv imagenet64-iter-1600000-model-ema.th pretrained/imagenet64/model-ema.th

mkdir pretrained/ffhq256
wget https://openaipublic.blob.core.windows.net/very-deep-vaes-assets/vdvae-assets/ffhq256-iter-1700000-model-ema.th
mv ffhq256-iter-1700000-model-ema.th pretrained/ffhq256/model-ema.th
