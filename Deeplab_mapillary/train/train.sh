export PYTHONPATH=$PYTHONPATH:../models/research/:../models/research/slim
mkdir -p logs/
now=$(date +"%Y%m%d_%H%M%S")
python ../models/research/deeplab/train.py \
    --logtostderr \
    --initialize_last_layer=False \
    --last_layers_contain_logits_only=True \
    --training_number_of_steps=150000 \
    --train_split="train" \
    --model_variant="xception_65" \
    --atrous_rates=6 \
    --atrous_rates=12 \
    --atrous_rates=18 \
    --output_stride=16 \
    --decoder_output_stride=4 \
    --train_crop_size=500 \
    --train_crop_size=500 \
    --train_batch_size=16 \
    --num_clones=4 \
    --dataset="mapillary" \
    --save_summaries_secs=60 \
    --base_learning_rate=0.001 \
    --learning_rate_decay_step=200 \
    --weight_decay=0.000015 \
    --fine_tune_batch_norm=True \
    --save_summaries_images=True \
    --tf_initial_checkpoint="/mnt/fcav/self_training_segmentation/Deeplab_mapillary/pretrained_model/trainlogs/model.ckpt-50000" \
    --train_logdir="/mnt/fcav/self_training_segmentation/Deeplab_mapillary/trainlogs/" \
    --dataset_dir="/mnt/fcav/self_training_segmentation/Deeplab_mapillary/images/tfrecord/" 2>&1 | tee logs/train_$now.txt &
    #--tf_initial_checkpoint="pretrained_model/deeplabv3_pascal_trainval/model.ckpt" \
