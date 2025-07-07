mkdir -p dataset/GRAB_HANDS/diffh2o_representation_full
mkdir -p dataset/GRAB_HANDS/diffh2o_representation_grasp
mkdir -p dataset/GRAB_HANDS/diffh2o_representation_interaction

echo -e "Downloading our processed GRAB representations"
curl -L "https://dataset.ait.ethz.ch/downloads/diffh2o/diffh2o_representation_full.zip" -o "dataset/diffh2o_representation_full.zip"
unzip -o dataset/diffh2o_representation_full.zip -d dataset/GRAB_HANDS
curl -L "https://dataset.ait.ethz.ch/downloads/diffh2o/diffh2o_representation_grasp.zip" -o "dataset/diffh2o_representation_grasp.zip"
unzip -o dataset/diffh2o_representation_grasp.zip -d dataset/GRAB_HANDS
curl -L "https://dataset.ait.ethz.ch/downloads/diffh2o/diffh2o_representation_interaction.zip" -o "dataset/diffh2o_representation_interaction.zip"
unzip -o dataset/diffh2o_representation_interaction.zip -d dataset/GRAB_HANDS

unzip -o dataset/GRAB_HANDS/text_annotations.zip -d dataset/GRAB_HANDS
rm -rf dataset/GRAB_HANDS/text_annotations.zip

echo -e "Cleaning\n"
rm dataset/diffh2o_representation_full.zip
rm dataset/diffh2o_grasp.zip
rm dataset/diffh2o_interaction.zip
rm dataset/GRAB_HANDS/__MACOSX

echo -e "Downloading glove (in use by the evaluators, not by DiffH2O itself)"
gdown --fuzzy https://drive.google.com/file/d/1cmXKUT31pqd7_XpJAiWEo1K81TMYHA5n/view?usp=sharing
rm -rf glove

unzip glove.zip
echo -e "Cleaning\n"
rm glove.zip

echo -e "Downloading done!"