threshold=0.9
model_calib=False

wbit=8
abit=8

arch="warp"

python -m torch.distributed.launch --nproc_per_node=1 --master_port 47770 ptq.py \
    --data_dir /share/seo/multiface/m--20180426--0000--002643814--GHS \
    --krt_dir /share/seo/multiface/m--20180426--0000--002643814--GHS/KRT \
    --framelist_train /share/seo/multiface/m--20180426--0000--002643814--GHS/frame_list.txt \
    --framelist_test /share/seo/multiface/m--20180426--0000--002643814--GHS/frame_list.txt \
    --model_ckpt "./pretrained_model/002643814/${arch}/best_model.pth" \
    --arch ${arch} \
    --n_worker 1 \
    --wbit ${wbit} \
    --abit ${abit} \
    --tau ${threshold} \
    --num_samples 512 \
    --lr 1e-4 \
    --model_calib ${model_calib} \
    --train_batch_size 8 \
    --result_path "./runs/experiment_002643814/PTQ_${arch}_w${wbit}a${wbit}_shape_batch_wise_mask_tau${threshold}_model_calib${model_calib}/"
