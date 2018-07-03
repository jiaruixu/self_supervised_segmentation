export PYTHONPATH=$PYTHONPATH:/home/jiarui/models/research/slim:/home/jiarui/git/Deeplab_KITTI
python vis.py \
    --logtostderr \
    --vis_split="val" \
    --image_side="left" \
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
    --checkpoint_dir="pretrained_model/" \
    --vis_logdir="leftvis/" \
    --dataset_dir="/home/jiarui/git/Deeplab_KITTI/KITTI_to_tfrecord/leftTfrecord" &
