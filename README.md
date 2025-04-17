# cGAN-Diagnose-Diabetic retinopathy
Diagnose Diabetic retinopathy (DR) using synthetic OCT-A images from Fundus images. 
The Dataset is available in the following: 
https://zenodo.org/records/6476639 [1]

Proposed Methodology:
The fundus imaging (no depth information of retinal vasculature) and OCT (no vasculature) in DR diagnosis, we propose a first of the kind model, combining the two modalities i. (original) fundus images ii. (GAN synthesized) OCT-A images to diagnose DR. The fundus imaging offers broader visualization of Optic Disc, macula whereas OCTA offers depth resolved information of Blood Vessels, both stand complementary in the diagnosis of DR. We feed the fundus images and BVAC synthesized OCT-A images to
2 ResNet 50 CNNs with a merged Fully Connected layer for DR diagnosis.

The following steps are executed to generate synthetic OCT-A image pair from Fundus image and diagnose DR.
1. pre-processing steps
2. Pipeline to create dataset for train, test of Fundus and OCT-A images
3. Generator: Improved U-Net with SE block
4. Encoder stage of U-Net: customized threshold-based Squeeze and excitaion (SE) block 
5. Loss: Generator loss, discriminator loss
6. Train a model
7. Develop an Web based application (APP) to predict OCT-A pair from Fundus image

Reference:
[1]. Coronado I, Pachade S, Trucco E, Abdelkhaleq R, Yan J, Salazar-Marioni S, Jagolino-Cole A, Bahrainian M, Channa R, Sheth SA, Giancardo L. Synthetic OCT-A blood vessel maps using fundus images and generative adversarial networks. Sci Rep 2023;13:15325. https://doi.org/10.1038/s41598-023-42062-9.
