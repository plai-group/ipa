wget https://people.eecs.berkeley.edu/~tinghuiz/projects/pix2pix/datasets/edges2shoes.tar.gz
tar -xvf edges2shoes.tar.gz

# put into subfolders
mkdir edges2shoes/train/all
mkdir edges2shoes/val/all
mv edges2shoes/train/*.jpg edges2shoes/train/all/
mv edges2shoes/val/*.jpg edges2shoes/val/all/
