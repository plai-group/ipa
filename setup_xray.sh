cd chest-xrays
python batch_download_zips.py
bash unpack.sh
python downsample.py
python split_train_test.py
