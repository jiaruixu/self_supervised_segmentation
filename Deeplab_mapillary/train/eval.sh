export PYTHONPATH=$PYTHONPATH:../models/research/:../models/research/slim
python ../models/research/deeplab/eval.py \
    --logtostderr \
    --eval_split="val" \
    --model_variant="xception_65" \
    --atrous_rates=6 \
    --atrous_rates=12 \
    --atrous_rates=18 \
    --output_stride=16 \
    --decoder_output_stride=4 \
    --eval_crop_size=1025 \
    --eval_crop_size=2049 \
    --dataset="mapillary" \
    --checkpoint_dir="trained_model/trainlog/" \
    --eval_logdir="eval/" \
    --dataset_dir="../tfrecord/tfrecord/" &
