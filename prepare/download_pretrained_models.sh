echo -e "Downloading pretrained models"
gdown "https://drive.google.com/uc?id=1BptobrakVS3NoUvkL_VVvBp6P87_XMqo" -O "save/"
unzip save/diffh2o_grasp.zip -d save/
gdown "https://drive.google.com/uc?id=1CgFZCKJT5_vxKGXCaBXBzpn1VRUkEFUN" -O "save/"
unzip save/diffh2o_full.zip -d save/
gdown "https://drive.google.com/uc?id=1cqyHe8MIzsOW3HS0TtakMWAj3eNdrZev" -O "save/"
unzip save/diffh2o_full_detailed.zip -d save/

echo -e "Cleaning\n"
rm save/diffh2o_grasp.zip
rm save/diffh2o_full.zip
rm save/diffh2o_full_detailed.zip

echo -e "Downloading done!"