import torch
import numpy as np
from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images
import torch.nn.functional as F
from PIL import Image
from prior_depth_anything.utils import (
    depth2disparity,
    disparity2depth
)
from vggt.utils.geometry import unproject_depth_map_to_point_map
from vggt.utils.pose_enc import pose_encoding_to_extri_intri


def project_point_map_to_depth_map(
        point_map: torch.Tensor, extrinsics_cam: torch.Tensor, intrinsics_cam: torch.Tensor, size
) -> torch.Tensor:
    depth_map_list = []
    for frame_idx in range(point_map.shape[0]):
        curr_depth_map = world_coords_points_to_depth(
            point_map[frame_idx], extrinsics_cam[frame_idx], intrinsics_cam[frame_idx], size
        )

        depth_map_list.append(curr_depth_map)

    depth_map_array = torch.stack(depth_map_list)

    return depth_map_array


def world_coords_points_to_depth(
        point_map: torch.Tensor,
        extrinsic: torch.Tensor,
        intrinsic: torch.Tensor,
        size,
        eps=1e-8,
) -> torch.Tensor:
    R = extrinsic[:3, :3]
    T = extrinsic[:3, 3:]

    cam_point_map = point_map @ R.T + T.T
    points_3d_homo = cam_point_map.T
    projected_points = intrinsic @ points_3d_homo

    if size[0] * size[1] == point_map.shape[0]:
        depth_map = projected_points[2, :].reshape(size)
    else:
        projected_points /= projected_points[2, :]

        height, width = size
        depth_map = torch.full((height, width), torch.inf).to(point_map.device)
        u, v = torch.round(projected_points[0, :]).to(torch.int), torch.round(projected_points[1, :]).to(torch.int)
        valid_mask = (0 <= u) & (u < width) & (0 <= v) & (v < height)

        for i in range(len(u)):
            if valid_mask[i] and cam_point_map[i, 2] < depth_map[v[i], u[i]]:
                depth_map[v[i], u[i]] = cam_point_map[i, 2]

        depth_map[depth_map > 1e5] = 0
    return depth_map


def depth_map_to_3D_points(depth_map, img_rgb, focal_length, principal_point=None):
    """
    Convert a depth map to 3D points.

    Args:
    - depth_map (numpy.ndarray): Depth map.
    - focal_length (float): Focal length of the camera.
    - principal_point (tuple): Principal point of the camera.

    Returns:
    - numpy.ndarray: 3D points.
    """
    height, width = depth_map.shape
    if principal_point is None:
        u0 = width / 2
        v0 = height / 2
    else:
        u0, v0 = principal_point
    points = np.zeros((height, width, 3))
    colors = np.zeros((height, width, 3))
    for v in range(height):
        for u in range(width):
            z = depth_map[v, u]
            x = (u - u0) * z / focal_length
            y = (v - v0) * z / focal_length
            points[v, u] = [x, y, z]
            colors[v, u] = img_rgb[v, u]

    return points, colors


def write_ply(filename, points, colors):
    """
    Write points and colors to a PLY file.

    :param filename: The name of the PLY file to write.
    :param points: A numpy array of shape (N, 3) containing the 3D points.
    :param colors: A numpy array of shape (N, 3) containing the RGB colors.
    """
    # Ensure colors are in the range [0, 255]
    if colors.max() <= 1.0:
        colors = (colors * 255).astype(np.uint8)

    # Create the PLY header
    header = f"ply\nformat ascii 1.0\nelement vertex {points.shape[0]}\nproperty float x\nproperty float y\nproperty float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n"
    # Write the header and data to the file
    with open(filename, 'w') as ply_file:
        ply_file.write(header)
        for point, color in zip(points, colors):
            ply_file.write(f"{point[0]} {point[1]} {point[2]} {color[0]} {color[1]} {color[2]}\n")
    print(f"Saved point cloud to {filename}")

if __name__ == '__main__':
    from matplotlib import pyplot as plt
    from pathlib import Path
    # DP_image_folder = Path("assets/sample-table")
    DP_image_folder = Path("assets/sample-bottle")
    # DP_image_folder = Path("/d_disk/Desktop/DemoBottle/images-100")
    FPS_images = DP_image_folder.glob("*.png")
    image_names = sorted(FPS_images)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # bfloat16 is supported on Ampere GPUs (Compute Capability 8.0+)
    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16

    # Initialize vggt.
    print("Initialize VGGT and load the pretrained weights.")
    vggt = VGGT().to(device)
    ckpt_path = '/dataset/wangzh/omni_dc/ckpts/vggt/model.pt'
    vggt = VGGT.from_pretrained("facebook/VGGT-1B").to(device)
    print("VGGT loaded!")

    # Initialize prior-depth-anything module.
    from prior_depth_anything.plugin import PriorDARefiner, PriorDARefinerMetrics

    Refiner = PriorDARefiner(device=device)

    with torch.no_grad():
        ########## Depth-Estimation stage.
        # depth_name = 'assets/sample-2/gt_depth.png'
        images = load_and_preprocess_images(image_names).to(device)

        # Predict attributes including cameras, depth maps, and point maps.
        with torch.cuda.amp.autocast(dtype=dtype):
            predictions = vggt(images)
    predictions['refined_depth_np'] = {}
    for idx, image_name in enumerate(image_names):
        ########## Refine stage.
        # Load gt for evaluation.
        # gt_depth = torch.from_numpy(
        #     np.asarray(Image.open(depth_name)).astype(np.float32)).to(device)
        priorda_image = torch.from_numpy(np.asarray(Image.open(image_name)).astype(np.uint8))

        ### Refine depth
        depth_map, depth_conf = predictions['depth'].squeeze()[idx], predictions['depth_conf'].squeeze()[idx]
        refined_depth, meview_depth_map = Refiner.predict(
            image=priorda_image, depth_map=depth_map.squeeze(), confidence=depth_conf.squeeze()
        )       # 1920 x 1080 Image takes 15.1GB GMem, which is suppose to be the maximum image size.

        # predictions['refined_depth'] = F.interpolate(
        #     refined_depth[None, None, ...], size=(depth_map.shape[-3], depth_map.shape[-2]),
        #     mode='bilinear', align_corners=True
        # )
        predictions['refined_depth_np'][idx] = refined_depth.squeeze().cpu().numpy()
        # raw_metrics, refined_metrics = Refiner.raw_refined_metrics(
        #     gt_depth=gt_depth, raw_depth=meview_depth_map, refined_depth=refined_depth
        # )
        pose_enc = predictions['pose_enc']
        # extrinsic, intrinsic = pose_encoding_to_extri_intri(pose_enc, images.shape[-2:])
        extrinsic, intrinsic = pose_encoding_to_extri_intri(
            pose_enc,
            image_size_hw=refined_depth.shape[-2:]
        )
        focal = (intrinsic[0, :, 0, 0] + intrinsic[0, :, 1, 1]).mean().cpu().numpy() / 2
        pts, col = depth_map_to_3D_points(
            refined_depth.squeeze().cpu().numpy(), priorda_image, focal
        )
        write_ply(
            filename=f"output_{image_name.stem}.ply",
            points=pts.reshape(-1, 3),
            colors=col.reshape(-1, 3).astype(np.uint8)
        )
        # TODO: add extrinsics to output the point cloud, so that they can be stitched together.
        ### Refine point_map.
        # world_points, world_points_conf = predictions['world_points'], predictions['world_points_conf']
    pose_enc = predictions['pose_enc'][0][idx]
    # extrinsic, intrinsic = pose_encoding_to_extri_intri(pose_enc, images.shape[-2:])
    extrinsic, intrinsic = pose_encoding_to_extri_intri(pose_enc, priorda_image.shape[-2:])
    # depth_by_project = project_point_map_to_depth_map(
    #     world_points.view(-1, 3).unsqueeze(0), extrinsics_cam=extrinsic.squeeze(0),
    #     intrinsics_cam=intrinsic.squeeze(0), size=images.shape[-2:]
    # )
    #
    # refined_projected, meview_depth_by_project = Refiner.predict(
    #     image=priorda_image, depth_map=depth_by_project.squeeze(), confidence=world_points_conf.squeeze())
    # inview_refined_projected = F.interpolate(
    #     refined_depth[None, None, ...], size=(depth_map.shape[-3], depth_map.shape[-2]),
    #     mode='bilinear', align_corners=True
    # ).squeeze()
    refined_world_points = unproject_depth_map_to_point_map(
        refined_depth.squeeze(), extrinsic.squeeze(0), intrinsic.squeeze(0))
    predictions['refined_points'] = refined_world_points

        # TODO: Add the Point Map regarding the dense depth map.
