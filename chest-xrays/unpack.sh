mkdir all-images

for i in "01" "02" "03" "04" "05" "06" "07" "08" "09" "10" "11" "12"
do
  echo "unpacking images_${i}.tar.gz"
  tar -xf "images_${i}.tar.gz"
  mv images/* all-images
  rm -r images
done
