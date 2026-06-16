echo "Starting training smile classifier without augmentation"
python train_classifier.py 31 smile

echo "Starting training smile classifier with augmentation"
python train_classifier.py 31 smile --augment


echo "Starting training male classifier without augmentation"
python train_classifier.py 20 male

echo "Starting training male classifier with augmentation"
python train_classifier.py 20 male --augment
