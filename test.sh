python test.py \
--model_name FIQA_EdgeNeXt_XXS \
--model_weights_file ckpts/EdgeNeXt_XXS_checkpoint.pt \
--image_file demo_images/z06399.png \
--image_size 356 \
--gpu_ids 0

python test.py \
--model_name FIQA_Swin_B \
--model_weights_file /data/sunwei_data/ModelFolder/VQualA_FIQA/swin_linear_mse_plcc_with_faceimages_20w_FIQA/Swin_B_plus_checkpoint.pt \
--image_file demo_images/z06399.png \
--image_size 448 \
--gpu_ids 0