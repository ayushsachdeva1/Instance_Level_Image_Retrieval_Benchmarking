from pathlib import Path
import argparse
import numpy as np
import matplotlib.cm as cm
import torch
import pandas as pd
import json

from collections import defaultdict

from models.matching import Matching
from models.utils import (compute_pose_error, compute_epipolar_error,
                          estimate_pose, make_matching_plot,
                          error_colormap, AverageTimer, pose_auc, read_image,
                          rotate_intrinsics, rotate_pose_inplane,
                          scale_intrinsics)

torch.set_grad_enabled(False)

def main():
    parser = argparse.ArgumentParser(
        description='Image pair matching and pose evaluation with SuperGlue',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument(
        '--input_dir', type=str, default='/scratch/as216/amur/train_small',
        help='Path to the directory that contains the images')
    parser.add_argument(
        '--output_dir', type=str, default='test_results/',
        help='Path to the directory in which the .npz results and optionally,'
             'the visualization images are written')
    parser.add_argument(
        '--resize', type=int, nargs='+', default=[640, 480],
        help='Resize the input image before running inference. If two numbers, '
             'resize to the exact dimensions, if one number, resize the max '
             'dimension, if -1, do not resize')
    parser.add_argument(
        '--superglue', choices={'indoor', 'outdoor'}, default='indoor',
        help='SuperGlue weights')
    parser.add_argument(
        '--max_keypoints', type=int, default=1024,
        help='Maximum number of keypoints detected by Superpoint'
             ' (\'-1\' keeps all keypoints)')
    parser.add_argument(
        '--keypoint_threshold', type=float, default=0.005,
        help='SuperPoint keypoint detector confidence threshold')
    parser.add_argument(
        '--nms_radius', type=int, default=4,
        help='SuperPoint Non Maximum Suppression (NMS) radius'
             ' (Must be positive)')
    parser.add_argument(
        '--sinkhorn_iterations', type=int, default=20,
        help='Number of Sinkhorn iterations performed by SuperGlue')
    parser.add_argument(
        '--match_threshold', type=float, default=0.2,
        help='SuperGlue match threshold')
    parser.add_argument(
        '--viz', action='store_true',
        help='Visualize the matches and dump the plots')
    parser.add_argument(
        '--show_keypoints', action='store_true',
        help='Plot the keypoints in addition to the matches')
    parser.add_argument(
        '--viz_extension', type=str, default='png', choices=['png', 'pdf'],
        help='Visualization file extension. Use pdf for highest-quality.')

    opt = parser.parse_args()
    print(opt)


    def pair_matching(name0, name1, timer):
        stem0, stem1 = Path(name0).stem, Path(name1).stem
        matches_path = output_dir / '{}_{}_matches.npz'.format(stem0, stem1)
        viz_path = output_dir / '{}_{}_matches.{}'.format(stem0, stem1, opt.viz_extension)
        rot0, rot1 = 0, 0
        image0, inp0, scales0 = read_image(input_dir / name0, device, [640, 480], rot0, False)
        image1, inp1, scales1 = read_image(input_dir / name1, device, [640, 480], rot1, False)
        if image0 is None or image1 is None:
            print('Problem reading image pair: {} {}'.format(
                input_dir / name0, input_dir / name1))
            exit(1)
        timer.update('load_image')
        pred = matching({'image0': inp0, 'image1': inp1})
        pred = {k: v[0].cpu().numpy() for k, v in pred.items()}
        kpts0, kpts1 = pred['keypoints0'], pred['keypoints1']
        matches, conf = pred['matches0'], pred['matching_scores0']
        timer.update('matcher')
        # Write the matches to disk.
        out_matches = {'keypoints0': kpts0, 'keypoints1': kpts1, 'matches': matches, 'match_confidence': conf}
        # np.savez(str(matches_path), **out_matches)

        valid = matches > -1
        mkpts0 = kpts0[valid]
        mkpts1 = kpts1[matches[valid]]
        mconf = conf[valid]

        if opt.viz:
            # Visualize the matches.
            color = cm.jet(mconf)
            text = [
                'SuperGlue',
                'Keypoints: {}:{}'.format(len(kpts0), len(kpts1)),
                'Matches: {}'.format(len(mkpts0)),
            ]
            if rot0 != 0 or rot1 != 0:
                text.append('Rotation: {}:{}'.format(rot0, rot1))

            # Display extra parameter info.
            k_thresh = matching.superpoint.config['keypoint_threshold']
            m_thresh = matching.superglue.config['match_threshold']
            small_text = [
                'Keypoint Threshold: {:.4f}'.format(k_thresh),
                'Match Threshold: {:.2f}'.format(m_thresh),
                'Image Pair: {}:{}'.format(stem0, stem1),
            ]

            make_matching_plot(
                image0, image1, kpts0, kpts1, mkpts0, mkpts1, color,
                text, viz_path, opt.show_keypoints,
                False, False, 'Matches', small_text)

            timer.update('viz_match')

        return out_matches


    device = 'cuda:2' if torch.cuda.is_available() else 'cpu'
    print('Running inference on device \"{}\"'.format(device))
    config = {
        'superpoint': {
            'nms_radius': opt.nms_radius,
            'keypoint_threshold': opt.keypoint_threshold,
            'max_keypoints': opt.max_keypoints
        },
        'superglue': {
            'weights': opt.superglue,
            'sinkhorn_iterations': opt.sinkhorn_iterations,
            'match_threshold': opt.match_threshold,
        }
    }

    matching = Matching(config).eval().to(device)

    # train_df = pd.read_csv("/scratch/as216/amur/reid_list_train.csv", header = None)
    # train_df.columns = ["id", "image"]

    # images1 = list(train_df[train_df['id'] == 0]['image'])
    # images2 = list(train_df[train_df['id'] == 250]['image'])
    # images3 = list(train_df[train_df['id'] == 256]['image'])
    # images4 = list(train_df[train_df['id'] == 138]['image'])

    # files = images1 + images2 + images3 + images4

    files = ["003827.jpg", "001387.jpg"]

    # Create the output directories if they do not exist already.
    input_dir = Path(opt.input_dir)
    print('Looking for data in directory \"{}\"'.format(input_dir))
    output_dir = Path(opt.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    print('Will write matches to directory \"{}\"'.format(output_dir))

    if opt.viz:
        print('Will write visualization images to', 'directory \"{}\"'.format(output_dir))


    timer = AverageTimer(newline=True)
    iter = 0

    query_map = defaultdict(list)

    # files = list(input_dir.glob("*.jpg"))
    for i in range(len(files)):
        for j in range(i+1, len(files)):
            
            out = pair_matching(files[i], files[j], timer)

            metric = np.sum(out["match_confidence"]*(out["match_confidence"] > 0.2))

            query_map[int(Path(files[i]).stem)].append((int(Path(files[j]).stem), metric))
            query_map[int(Path(files[j]).stem)].append((int(Path(files[i]).stem), metric))
            
            if j == i+1:
                timer.print('Finished pair {:5}'.format(iter))
            iter += 1

    final_json = []

    for elem in query_map.keys():
        dct = {}
        dct["query_id"] = elem

        lst = query_map[elem]
        lst.sort(key = lambda x: x[1], reverse=True)

        dct["ans_ids"] = [x[0] for x in lst]

        final_json.append(dct)
    
    with open("superglue_results_test.json", 'w') as f:
        json.dump(final_json, f)


if __name__ == '__main__':
    main()
