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
    --dataset="mapillary" \
    --colormap_type="mapillary" \
    --checkpoint_dir="trained_model/trainlog/" \
    --vis_logdir="vis/" \
    --dataset_dir="../tfrecord/tfrecord/" &
