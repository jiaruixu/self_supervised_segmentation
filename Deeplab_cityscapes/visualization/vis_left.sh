export PYTHONPATH=$PYTHONPATH:/home/jiarui/models/research/slim:/mnt/data/Deeplab_test/Deeplab_cityscapes
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
    --vis_crop_size=1025 \
    --vis_crop_size=2049 \
    --dataset="cityscapes" \
    --colormap_type="cityscapes" \
    --checkpoint_dir="pretrained_model/" \
    --vis_logdir="leftvis/" \
    --dataset_dir="../tfrecord_optical_flow/tfrecord" &
