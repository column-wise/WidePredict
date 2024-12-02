import argparse
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from attrdict import AttrDict

from sgan.data.loader import data_loader
from sgan.models import TrajectoryGenerator
from sgan.losses import displacement_error, final_displacement_error
from sgan.utils import relative_to_abs, get_dset_path

parser = argparse.ArgumentParser()
parser.add_argument('--model_path', type=str)
parser.add_argument('--num_samples', default=20, type=int)
parser.add_argument('--dset_type', default='test', type=str)

xdata, ydata = [], []
x1=[]
y1=[]
aa, bb = [], []
fig, ax = plt.subplots()
ln, = ax.plot([], [], 'ro')

def gen_dot():
    for i in range(0,len(x1)):
        newdot = [x1[i], y1[i]]
        yield newdot

def update_dot(newd):
    xdata.append(newd[0])
    ydata.append(newd[1])
    ln.set_data(xdata, ydata)
    return ln

def get_generator(checkpoint):
    args = AttrDict(checkpoint['args'])
    generator = TrajectoryGenerator(
        obs_len=args.obs_len,
        pred_len=args.pred_len,
        embedding_dim=args.embedding_dim,
        encoder_h_dim=args.encoder_h_dim_g,
        decoder_h_dim=args.decoder_h_dim_g,
        mlp_dim=args.mlp_dim,
        num_layers=args.num_layers,
        noise_dim=args.noise_dim,
        noise_type=args.noise_type,
        noise_mix_type=args.noise_mix_type,
        pooling_type=args.pooling_type,
        pool_every_timestep=args.pool_every_timestep,
        dropout=args.dropout,
        bottleneck_dim=args.bottleneck_dim,
        neighborhood_size=args.neighborhood_size,
        grid_size=args.grid_size,
        batch_norm=args.batch_norm)
    generator.load_state_dict(checkpoint['g_state'])
    generator.cuda()
    generator.train()
    return generator


def evaluate_helper(error, seq_start_end):
    sum_ = 0
    error = torch.stack(error, dim=1)

    for (start, end) in seq_start_end:
        start = start.item()
        end = end.item()
        _error = error[start:end]
        _error = torch.sum(_error, dim=0)
        _error = torch.min(_error)
        sum_ += _error
    return sum_


def evaluate(args, loader, generator, num_samples):
    ade_outer, fde_outer = [], []
    total_traj = 0
    with torch.no_grad():
        for batch in loader:
            batch = [tensor.cuda() for tensor in batch]
            (obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel,
             non_linear_ped, loss_mask, seq_start_end) = batch

            ade, fde = [], []
            total_traj += pred_traj_gt.size(1)

            for _ in range(num_samples):
                pred_traj_fake_rel = generator(
                    obs_traj, obs_traj_rel, seq_start_end
                )

                # print('obs_traj\n', obs_traj)
                # print('obs_traj_rel\n',obs_traj_rel)

                pred_traj_fake = relative_to_abs(
                    pred_traj_fake_rel, obs_traj[-1]
                )

                print('seg_start_end\n', seq_start_end)

                gt = pred_traj_gt[:, 3, :].data
                print(gt)
                input_a = obs_traj[:, 3, :].data
                out_a = pred_traj_fake[:, 3, :].data
                #aa = np.concatenate((input_a, gt), axis=0)
                #bb = np.concatenate((input_a, out_a), axis=0)
                aa = np.concatenate((input_a.cuda().data.cpu().numpy(), gt.cuda().data.cpu().numpy()), axis=0)
                bb = np.concatenate((input_a.cuda().data.cpu().numpy(), out_a.cuda().data.cpu().numpy()), axis=0)
                global x1, y1

                ax.set_xlim(np.min(aa[:,0] -1), np.max(aa[:,0]) +1)
                ax.set_ylim(np.min(aa[:,1] -1), np.max(aa[:,1]) +1)

                x1 = bb[:, 0]
                y1 = bb[:, 1]
                l = ax.plot(aa[:, 0], aa[:, 1], '.')
                ani = animation.FuncAnimation(fig, update_dot, frames=gen_dot, interval=50)
                plt.show()
                plt.close()

                ade.append(displacement_error(
                    pred_traj_fake, pred_traj_gt, mode='raw'
                ))
                fde.append(final_displacement_error(
                    pred_traj_fake[-1], pred_traj_gt[-1], mode='raw'
                ))

            ade_sum = evaluate_helper(ade, seq_start_end)
            fde_sum = evaluate_helper(fde, seq_start_end)

            ade_outer.append(ade_sum)
            fde_outer.append(fde_sum)
        ade = sum(ade_outer) / (total_traj * args.pred_len)
        fde = sum(fde_outer) / (total_traj)
        return ade, fde


def main(args):
    if os.path.isdir(args.model_path):
        filenames = os.listdir(args.model_path)
        filenames.sort()
        paths = [
            os.path.join(args.model_path, file_) for file_ in filenames
        ]
    else:
        paths = [args.model_path]

    for path in paths:
        checkpoint = torch.load(path)
        generator = get_generator(checkpoint)
        _args = AttrDict(checkpoint['args'])
        #path = get_dset_path(_args.dataset_name, args.dset_type)
        path = "D:/DeepSort/sgan/datasets/customDataset"
        _, loader = data_loader(_args, path)
        #batch = next(iter(loader))
        #print(batch)
        ade, fde = evaluate(_args, loader, generator, args.num_samples)
        print('Dataset: {}, Pred Len: {}, ADE: {:.2f}, FDE: {:.2f}'.format(
            _args.dataset_name, _args.pred_len, ade, fde))


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
