#!/usr/bin/env python

import argparse

from cosmo_vgg import models


def parse_args():
    parser = argparse.ArgumentParser(description='Cosmological encoder (2D/3D)')
    parser.add_argument('--mode', choices=['train', 'embed', 'fid'], default='train')

    # Data
    parser.add_argument('--data_dir', type=str, default='./data')
    parser.add_argument('--train_dir', type=str, default='./data/train')
    parser.add_argument('--gen_dir', type=str, default='./data/generated')
    parser.add_argument('--in_channels', type=int, default=1,
                        help='Number of input field channels (3D mode only)')
    parser.add_argument('--resolution', type=int, default=64,
                        help='Spatial resolution of input cube (cubic)')

    # 2D / 3D mode
    parser.add_argument('--twodim', action='store_true',
                        help='Run in 2D mode: thin D axis and encode per-slice')
    parser.add_argument('--thin_factor', type=int, default=8,
                        help='Factor to reduce D axis by in 2D mode (default: 8)')

    # Model
    parser.add_argument('--base_channels', type=int, default=32)
    parser.add_argument('--embed_dim', type=int, default=256)
    parser.add_argument('--attn_heads', type=int, default=8)

    # Training
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--warmup_steps', type=int, default=500)
    parser.add_argument('--regression_weight', type=float, default=0.1)
    parser.add_argument('--num_workers', type=int, default=4)

    # Augmentation
    parser.add_argument('--noise_std', type=float, default=0.05)
    parser.add_argument('--kspace_mask_prob', type=float, default=0.3)

    # I/O
    parser.add_argument('--checkpoint', type=str, default='encoder.pt')

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    if args.mode == 'train':
        models.train(
            data_dir=args.data_dir,
            checkpoint=args.checkpoint,
            resolution=args.resolution,
            in_channels=args.in_channels,
            base_channels=args.base_channels,
            embed_dim=args.embed_dim,
            attn_heads=args.attn_heads,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            weight_decay=args.weight_decay,
            warmup_steps=args.warmup_steps,
            regression_weight=args.regression_weight,
            num_workers=args.num_workers,
            noise_std=args.noise_std,
            kspace_mask_prob=args.kspace_mask_prob,
            twodim=args.twodim,
            thin_factor=args.thin_factor,
        )
    elif args.mode == 'embed':
        models.embed(
            checkpoint=args.checkpoint,
            data_dir=args.data_dir,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )
    elif args.mode == 'fid':
        models.compute_fid(
            checkpoint=args.checkpoint,
            train_dir=args.train_dir,
            gen_dir=args.gen_dir,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )
