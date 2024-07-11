"""
Post Training Quantization
"""

import argparse
import json
import os
import glob
import logging

import cv2
import torch
from dataset import Dataset
from models import DeepAppearanceVAE, WarpFieldVAE
from torch.utils.data import DataLoader
from ptq.ptq_trainer import PTQTrainer
from qlib.utils import str2bool

def main(args, camera_config, test_segment):
    local_rank = os.environ['LOCAL_RANK']
    local_rank = int(local_rank)
    torch.cuda.set_device(local_rank)
    device = torch.device('cuda', local_rank)

    # saving path
    if not os.path.isdir(args.result_path):
        os.makedirs(args.result_path, exist_ok=True)

    # logging
    logger = logging.getLogger('training')
    fileHandler = logging.FileHandler(args.result_path+"training.log")
    fileHandler.setLevel(0)
    logger.addHandler(fileHandler)

    streamHandler = logging.StreamHandler()
    streamHandler.setLevel(0)
    logger.addHandler(streamHandler)
    logger.root.setLevel(0)
    logger.info(args)


    dataset_train = Dataset(
        args.data_dir,
        args.krt_dir,
        args.framelist_train,
        args.tex_size,
        camset=None if camera_config is None else camera_config["train"],
        exclude_prefix=test_segment,
    )
    
    dataset_test = Dataset(
        args.data_dir,
        args.krt_dir,
        args.framelist_test,
        args.tex_size,
        camset=None if camera_config is None else camera_config["test"],
        valid_prefix=test_segment,
    )

    # fetch the calibration data
    rand = torch.utils.data.RandomSampler(dataset_train, num_samples=args.num_samples)
    sampler = torch.utils.data.BatchSampler(rand, batch_size=args.train_batch_size, drop_last=False)
    calib_loader = DataLoader(dataset_train, batch_sampler=sampler, num_workers=args.n_worker, pin_memory=True)

    texmean = cv2.resize(dataset_train.texmean, (args.tex_size, args.tex_size))
    texmin = cv2.resize(dataset_train.texmin, (args.tex_size, args.tex_size))
    texmax = cv2.resize(dataset_train.texmax, (args.tex_size, args.tex_size))
    texmean = torch.tensor(texmean).permute((2, 0, 1))[None, ...].to(device)
    texmin = torch.tensor(texmin).permute((2, 0, 1))[None, ...].to(device)
    texmax = torch.tensor(texmax).permute((2, 0, 1))[None, ...].to(device)
    texstd = dataset_train.texstd
    vertmean = torch.tensor(dataset_train.vertmean, dtype=torch.float32).view((1, -1, 3)).to(device)
    vertstd = dataset_train.vertstd

    if local_rank == 0:
        logger.info(f"# of calibration batches {len(calib_loader)}")
        logger.info(f"# of test samples: {len(dataset_test)}")

    n_cams = len(set(dataset_train.cameras).union(set(dataset_test.cameras)))
    if args.arch == "base":
        model = DeepAppearanceVAE(
            args.tex_size, args.mesh_inp_size, n_latent=args.nlatent, n_cams=n_cams
        ).to(device)
    elif args.arch == "res":
        model = DeepAppearanceVAE(
            args.tex_size,
            args.mesh_inp_size,
            n_latent=args.nlatent,
            res=True,
            n_cams=n_cams,
        ).to(device)
    elif args.arch == "warp":
        model = WarpFieldVAE(
            args.tex_size, args.mesh_inp_size, z_dim=args.nlatent, n_cams=n_cams
        ).to(device)
    elif args.arch == "non":
        model = DeepAppearanceVAE(
            args.tex_size,
            args.mesh_inp_size,
            n_latent=args.nlatent,
            res=False,
            non=True,
            n_cams=n_cams,
        ).to(device)
    elif args.arch == "bilinear":
        model = DeepAppearanceVAE(
            args.tex_size,
            args.mesh_inp_size,
            n_latent=args.nlatent,
            res=False,
            non=False,
            bilinear=True,
            n_cams=n_cams,
        ).to(device)
    else:
        raise NotImplementedError

    model = torch.nn.parallel.DistributedDataParallel(model, [local_rank], local_rank)

    if args.model_ckpt is not None:
        logger.info(f"loading checkpoint from {args.model_ckpt}")
        map_location = {"cuda:%d" % 0: "cuda:%d" % local_rank}
        model.load_state_dict(torch.load(args.model_ckpt, map_location=map_location))
        logger.info("Checkpoint loaded!")
    
    # visibility map
    loss_weight_mask = cv2.flip(cv2.imread(args.loss_weight_mask), 0)
    loss_weight_mask = loss_weight_mask / loss_weight_mask.max()
    loss_weight_mask = (
        torch.tensor(loss_weight_mask).permute(2, 0, 1).unsqueeze(0).float().to(device)
    )
    loss_weight_mask = loss_weight_mask[0,0].unsqueeze(0).unsqueeze(0)
    
    trainer = PTQTrainer(model, max_iter=100, dataloader=calib_loader, args=args, 
                         texmean=texmean, texstd=texstd, vertmean=vertmean, vertstd=vertstd, logger=logger)
    qmodel = trainer.fit()
    
    logger.info(qmodel)

    if local_rank == 0:
        torch.save(
            qmodel.state_dict(), os.path.join(args.result_path, "model.pth")
        )

    logger.info("PTQ Finished")


if __name__ == "__main__":
    torch.distributed.init_process_group(backend="nccl")
    parser = argparse.ArgumentParser(description="Process some integers.")
    parser.add_argument('--local-rank', type=int, default=0, help='Local rank for distributed run')
    parser.add_argument(
        "--train_batch_size", type=int, default=16, help="Training batch size"
    )
    parser.add_argument(
        "--val_batch_size", type=int, default=8, help="Validation batch size"
    )
    parser.add_argument(
        "--arch",
        type=str,
        default="base",
        help="Model architecture - base|warp|res|non|bilinear",
    )
    parser.add_argument(
        "--nlatent", type=int, default=256, help="Latent code dimension - 128|256"
    )
    parser.add_argument(
        "--lr", type=float, default=3e-4, help="Learning rate for training"
    )
    parser.add_argument(
        "--resolution",
        default=[2048, 1334],
        nargs=2,
        type=int,
        help="Rendering resolution",
    )
    parser.add_argument("--tex_size", type=int, default=1024, help="Texture resolution")
    parser.add_argument(
        "--mesh_inp_size", type=int, default=21918, help="Input mesh dimension"
    )
    parser.add_argument(
        "--epochs", type=int, default=50, help="Number of training epochs"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="/mnt/captures/zhengningyuan/m--20180226--0000--6674443--GHS",
        help="Directory to dataset root",
    )
    parser.add_argument(
        "--krt_dir",
        type=str,
        default="/mnt/captures/zhengningyuan/m--20180226--0000--6674443--GHS/KRT",
        help="Directory to KRT file",
    )
    parser.add_argument(
        "--loss_weight_mask",
        type=str,
        default="./loss_weight_mask.png",
        help="Mask for weighted loss of face",
    )
    parser.add_argument(
        "--framelist_train",
        type=str,
        default="/mnt/captures/zhengningyuan/m--20180226--0000--6674443--GHS/frame_list.txt",
        help="Frame list for training",
    )
    parser.add_argument(
        "--framelist_test",
        type=str,
        default="/mnt/captures/zhengningyuan/m--20180226--0000--6674443--GHS/frame_list.txt",
        help="Frame list for testing",
    )
    parser.add_argument(
        "--test_segment_config",
        type=str,
        default=None,
        help="Directory of expression segments for testing (exclude from training)",
    )
    parser.add_argument(
        "--lambda_verts", type=float, default=1, help="Multiplier of vertex loss"
    )
    parser.add_argument(
        "--lambda_screen", type=float, default=0, help="Multiplier of screen loss"
    )
    parser.add_argument(
        "--lambda_tex", type=float, default=1, help="Multiplier of texture loss"
    )
    parser.add_argument(
        "--lambda_kl", type=float, default=1e-2, help="Multiplier of KL divergence"
    )
    parser.add_argument(
        "--max_iter",
        type=int,
        default=200000,
        help="Maximum number of training iterations, overrides epoch",
    )
    parser.add_argument(
        "--log_every", type=int, default=1000, help="Interval of printing training loss"
    )
    parser.add_argument(
        "--val_every", type=int, default=5000, help="Interval of validating on test set"
    )
    parser.add_argument(
        "--val_num", type=int, default=500, help="Number of iterations for validation"
    )
    parser.add_argument(
        "--n_worker", type=int, default=8, help="Number of workers loading dataset"
    )
    parser.add_argument(
        "--pass_thres",
        type=int,
        default=50,
        help="If loss is x times higher than the previous batch, discard this batch",
    )
    parser.add_argument(
        "--result_path",
        type=str,
        default="./runs/experiment",
        help="Directory to output files",
    )
    parser.add_argument(
        "--model_ckpt", type=str, default=None, help="Model checkpoint path"
    )

    # for PTQ
    parser.add_argument(
        "--num_samples", type=int, default=1024, help="Number of iterations for validation"
    )
    parser.add_argument(
        "--wbit", type=int, default=8, help="weight precision"
    )
    parser.add_argument(
        "--abit", type=int, default=8, help="activation precision"
    )
    parser.add_argument(
        "--tau", type=float, default=0.7, help="Threshold of the standard deviation cutoff"
    )
    parser.add_argument(
        "--model_calib", type=str2bool, default=False, help="Flag of enable model wise calibration after layer-wise calib"
    )

    experiment_args = parser.parse_args()
    print(experiment_args)

    # load camera config
    subject_id = experiment_args.data_dir.split("--")[-2]
    camera_config_path = f"camera_configs/camera-split-config_{subject_id}.json"
    if os.path.exists(camera_config_path):
        print(f"camera config file for {subject_id} exists, loading...")
        f = open(camera_config_path, "r")
        camera_config = json.load(f)
        f.close()
    else:
        print(f"camera config file for {subject_id} NOT exists, generating...")
        # generate camera config based on downloaded data if not existed
        segments = [os.path.basename(x) for x in glob.glob(f"{experiment_args.data_dir}/unwrapped_uv_1024/*")]
        assert len(segments) > 0
        # select a segment to check available camera ids
        camera_ids = [os.path.basename(x) for x in glob.glob(f"{experiment_args.data_dir}/unwrapped_uv_1024/{segments[0]}/*")]
        camera_ids.remove('average')
        camera_config = {
            "full": {
                "train": camera_ids,
                "test": camera_ids,
                "visual": camera_ids[:2]
            }
        }    
        # save the config for future use
        os.makedirs("camera_configs", exist_ok=True)
        with open(camera_config_path, 'w') as f:
            json.dump(camera_config, f)

    camera_set = camera_config["full"]

    if experiment_args.test_segment_config is not None:
        f = open(experiment_args.test_segment_config, "r")
        test_segment_config = json.load(f)
        f.close()
        test_segment = test_segment_config["segment"]
    else:
        test_segment = ["EXP_ROM", "EXP_free_face"]

    main(experiment_args, camera_set, test_segment)
    
