# cGAN-Diagnose-Glaucoma
diagnose glaucoma using synthetic OCT-A images from Fundus images. 
Dataset: https://zenodo.org/records/6476639
1. pre-processing steps
2. Pipeline to create dataset for train, test of Fundus and OCT-A images
3. Generator: Improved U-Net with SE block and WHFA mechanism
4. Encoder stage of U-Net: customized threshold-based Squeeze and excitaion (SE) block 
5. Bottleck layer of U-Net: Wavelet based high-frequency attention (WHFA) mechanism
6. Loss: to be improved
7. Train a model
