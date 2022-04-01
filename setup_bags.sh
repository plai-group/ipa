# wget http://efrosgans.eecs.berkeley.edu/pix2pix/datasets/edges2handbags.tar.gz
tar -xvf edges2handbags.tar.gz

# put into subfolders
mkdir edges2handbags/train/all
mkdir edges2handbags/val/all
mv edges2handbags/train/0*.jpg edges2handbags/train/all/  # horrible way to cope with this error: `/usr/bin/mv: Argument list too long`
mv edges2handbags/train/10*.jpg edges2handbags/train/all/
mv edges2handbags/train/11*.jpg edges2handbags/train/all/
mv edges2handbags/train/12*.jpg edges2handbags/train/all/
mv edges2handbags/train/13*.jpg edges2handbags/train/all/
mv edges2handbags/train/14*.jpg edges2handbags/train/all/
mv edges2handbags/train/15*.jpg edges2handbags/train/all/
mv edges2handbags/train/16*.jpg edges2handbags/train/all/
mv edges2handbags/train/17*.jpg edges2handbags/train/all/
mv edges2handbags/train/18*.jpg edges2handbags/train/all/
mv edges2handbags/train/19*.jpg edges2handbags/train/all/
mv edges2handbags/train/2*.jpg edges2handbags/train/all/
mv edges2handbags/train/3*.jpg edges2handbags/train/all/
mv edges2handbags/train/4*.jpg edges2handbags/train/all/
mv edges2handbags/train/5*.jpg edges2handbags/train/all/
mv edges2handbags/train/6*.jpg edges2handbags/train/all/
mv edges2handbags/train/7*.jpg edges2handbags/train/all/
mv edges2handbags/train/8*.jpg edges2handbags/train/all/
mv edges2handbags/train/9*.jpg edges2handbags/train/all/
mv edges2handbags/val/*.jpg edges2handbags/val/all/
