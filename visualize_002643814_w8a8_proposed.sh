export CUDA_VISIBLE_DEVICES=0
batch_size=24
wbit=8
abit=8

viewID=400012
arch="warp"
report_psnr=False
save_tensor=True

kl_lambda=1.0

model_path="./runs/experiment_002183941/PTQ_warp_w8a8_shape_batch_wise_mask_tau0.05_model_calibFalse/model.pth"
sample_method="W${wbit}A${abit}"
calib_method="W8A8_POCA"
merged_video_name="merged_output_w8a8_POCA.mp4"
video_path="./runs/experiment_002643814/${viewID}/videos_${arch}_w${wbit}a${wbit}_poca/"
result_path="./runs/experiment_002643814/${viewID}/visual_${arch}_w${wbit}a${wbit}_poca/"
err_path="./runs/results_002643814/${viewID}/error_${arch}_w${wbit}a${wbit}_poca/"
tensor_path="./runs/results_002643814/results_002643814/${viewID}/pred_tensor_${arch}_w${wbit}a${wbit}_poca/"

for i in {1..3}
do
python -m torch.distributed.launch --nproc_per_node=1 --master_port 47778 visualize.py \
    --data_dir /share/seo/multiface/m--20180426--0000--002643814--GHS \
    --krt_dir /share/seo/multiface/m--20180426--0000--002643814--GHS/KRT \
    --framelist_test /share/seo/multiface/m--20180426--0000--002643814--GHS/frame_list.txt \
    --result_path ${result_path} \
    --video_path ${video_path} \
    --err_path ${err_path} \
    --tensor_path ${tensor_path} \
    --err_file "frame_wise_error_${i}.csv" \
    --test_segment "/home/jm2787/multiface-ptq/test_segments_002643814/mini_test_segment_group${i}.json" \
    --lambda_screen 1 \
    --model_path  ${model_path} \
    --camera_config "/home/jm2787/multiface-ptq/camera_configs/camera-split-config_002643814.json" \
    --camera_setting "full" \
    --arch ${arch} \
    --val_batch_size ${batch_size} \
    --sample_method ${sample_method} \
    --calib_method ${calib_method} \
    --wbit ${wbit} \
    --abit ${abit} \
    --report_psnr ${report_psnr} \
    --save_tensor ${save_tensor} \
    --n_worker 1
done

cd ${video_path}; find *.mp4 | sed 's:\ :\\\ :g'| sed 's/^/file /' > fl.txt; ffmpeg -f concat -i fl.txt -c copy ${merged_video_name}; rm fl.txt; cd ..
