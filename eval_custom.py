import argparse
from datetime import datetime
from pathlib import Path
from PIL import Image
import gin
import numpy as np
import torch

# Set Torch Hub Cache to /d_disk/torch_hub
dp_torch_hub = Path('/d_disk/torch_hub')
torch.hub.set_dir(dp_torch_hub.resolve().as_posix())
import os

os.environ['NUMEXPR_MAX_THREADS'] = '12'

from tqdm import tqdm
from multi_slam.MultiTrajectory import MultiTrajectory
from multi_slam.fullsystem import FullSystem
from multi_slam.locnet import LocNet


def get_image_list(dp_images):
    # find all the png and jpb without . or _ as starters
    list_images = [fp for fp in dp_images.glob('*.png') if not fp.stem.startswith(('.', '_'))]
    if not list_images:
        list_images = [fp for fp in dp_images.glob('*.jpg') if not fp.stem.startswith(('.', '_'))]

    return list_images


def read_image_into_cuda_tensor(fp_image, downscale_param=None):
    image = Image.open(fp_image)
    if downscale_param is not None:
        image = image.resize(downscale_param)
    image = torch.from_numpy(np.array(image)[..., :3]).permute(2, 0, 1).float().cuda()
    return image


@torch.no_grad()
def main(dp_path):
    torch.manual_seed(1234)
    np.random.seed(1234)

    pred_mt = MultiTrajectory("Estimated")

    twoview_system = LocNet().cuda().eval()
    twoview_system.load_weights("twoview.pth")

    vo_system = LocNet().cuda().eval()
    vo_system.load_weights("vo.pth")

    model = FullSystem(vo_system, twoview_system)

    start_time = datetime.now()

    name = dp_path.name
    list_images = get_image_list(dp_path)
    sample_img = Image.open(list_images[0])
    im_w_ori, im_h_ori = sample_img.size
    im_h_target, im_w_target = (400, 768)

    sample_img = sample_img.resize((im_w_target, im_h_target))
    im_w, im_h = sample_img.size
    model.add_new_video(name, len(list_images), (im_h, im_w))
    intrinsics = np.array([im_w, im_w, im_w // 2, im_h // 2])
    intrinsics = torch.as_tensor(intrinsics, dtype=torch.float, device='cuda')
    tstamp = 0
    for fp_image in tqdm(list_images):
        image = read_image_into_cuda_tensor(fp_image, downscale_param=(im_w_target, im_h_target))
        model.insert_frame(image, intrinsics, tstamp)
        tstamp += 1
    model.complete_video()
    model.backend_update(iters=10)

    results = model.terminate()
    end_time = datetime.now()

    base_dir = Path("our_predictions") / name
    base_dir.mkdir(exist_ok=True, parents=True)

    for scene_name, tstamp, pred_pose in results:
        pred_mt.insert(scene_name, tstamp, pred_pose)

    MultiTrajectory.plot_both(pred_mt, pred_mt, save_dir=base_dir)

    rmse_tr_err, rot_err, recalls = MultiTrajectory.error(pred_mt, pred_mt)
    text = f'Err (t): {rmse_tr_err:.03f}m | Err (R): {rot_err:.01f} deg | Recall {recalls} | {end_time - start_time}'
    print(text)
    (base_dir / "results.txt").write_text(text)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--folder', help='folder of images', required=True)
    args = parser.parse_args()

    gconfigs = [next(iter(Path('gconfigs').rglob(g)), None) for g in (["model/base.gin", "fullsystem.gin"])]
    assert all(gconfigs)
    gin.parse_config_files_and_bindings(gconfigs, [])

    main(Path(args.folder).resolve().absolute())

