export PYTHONPATH=$PYTHONPATH:../models/research/:../models/research/slim
python ../models/research/deeplab/vis.py \
    --logtostderr \
    --vis_split="val" \
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
    --checkpoint_dir="/mnt/fcav/self_training_segmentation/Deeplab_playing_for_data/trainlogs/" \
    --vis_logdir="/mnt/fcav/self_training_segmentation/Deeplab_playing_for_data/visualization/" \
    --dataset_dir="/mnt/fcav/self_training_segmentation/Deeplab_playing_for_data/tfrecord/" &
