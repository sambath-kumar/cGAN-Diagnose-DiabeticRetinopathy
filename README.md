# cGAN-Diagnose-Diabetic retinopathy
diagnose Diabetic retinopathy using synthetic OCT-A images from Fundus images. 
Dataset: https://zenodo.org/records/6476639
1. pre-processing steps
2. Pipeline to create dataset for train, test of Fundus and OCT-A images
3. Generator: Improved U-Net with SE block and WHFA mechanism
4. Encoder stage of U-Net: customized threshold-based Squeeze and excitaion (SE) block 
5. Loss: Generator loss, discriminator loss
6. Train a model
