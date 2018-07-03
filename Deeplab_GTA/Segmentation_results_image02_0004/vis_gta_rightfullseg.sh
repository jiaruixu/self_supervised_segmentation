export PYTHONPATH=$PYTHONPATH:/home/jiarui/models/research/:/home/jiarui/models/research/slim
python /home/jiarui/models/research/deeplab/vis_gta_fullseg.py \
    --logtostderr \
    --vis_split="val" \
    --image_side="right" \
    --model_variant="xception_65" \
    --atrous_rates=6 \
    --atrous_rates=12 \
    --atrous_rates=18 \
    --output_stride=16 \
    --decoder_output_stride=4 \
    --vis_crop_size=1081 \
    --vis_crop_size=1921\
    --dataset="cityscapes" \
    --colormap_type="cityscapes" \
    --checkpoint_dir="train/" \
    --vis_logdir="rightvis_fullseg/" \
    --dataset_dir="/home/jiarui/git/Deeplab_GTA/GTA_to_tfrecord_image02_0004/rightTfrecord" &
