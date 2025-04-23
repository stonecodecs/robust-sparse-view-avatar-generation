import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import proj3d
import json
import numpy as np
import pickle
import argparse
import os
from preprocessing import get_central_position


def visualize_camera_poses(
    camera_extrinsics,
    output_path=None,
    additional_transform=None,
    timestep=None,
    image_dir="mvset/mvhumannet_format_dir/images_lr",
    arrow_length=1,
    w2c=False,
):
    """
    Visualizes camera poses in 3D space. If timestep is provided with the correct dataset structure, image preview will be available.

    NOTE: make sure that extrinsics have been scaled by `camera_scale` before plotting.
        If timestep is provided, then the plot will be interactive with the images of the current timestep.
        Also, in MVHumanNet format, "1_CC32871A004.png" is an example key for the camera in camera_extrinsics.json.
        Internally, this is hardcoded to crop out "CC32871A004" and then find the timestep within it (if provided).

    Args:
        camera_extrinsics (dict): Dictionary containing camera poses and parameters
        output_path (str): Path where to save the visualization image
    """
    # Create 3D plot
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    ax.mouse_init() # for interactivemouse rotation
    
    # Store scatter points and their corresponding names
    positions = []
    scatter_points = []
    camera_names = []
    
    # Create a new axes for the hover image
    # hover_ax = fig.add_axes([0.8, 0.8, 0.2, 0.2])  # [left, bottom, width, height]
    # hover_ax.axis('off')
    # hover_im = hover_ax.imshow(np.zeros((100,100,3)))  # Placeholder image
    # hover_ax.set_visible(False)

    hover_ax = None
    current_scatter = None
    
    # Determine and process camera format
    if 'frames' in camera_extrinsics:
        # already in c2w format, already scaled with 'camera_scale'
        camera_list = camera_extrinsics['frames']
        for frame in camera_list:
            # Extract from 4x4 transform matrix
            transform_matrix = np.array(frame['transform_matrix'])
            R = transform_matrix[:3, :3]  # Rotation (world to camera)
            t = transform_matrix[:3, 3]   # Translation (world to camera)
            
            # Compute camera position in world coordinates
            if w2c:
                camera_pos = -R.T @ t # Equivalent to inv(transform_matrix)[:3, 3]
            else:
                camera_pos = t  
            # NOTE: 't' is POST-camera scaling
            
            # Apply additional transform if specified
            if additional_transform is not None:
                R = additional_transform @ R
                
            positions.append(camera_pos)
            img_name = frame['file_path']
            
            # Plot camera position
            scatter = ax.scatter(camera_pos[0], camera_pos[1], camera_pos[2],
                               c='blue', marker='o', picker=5, s=100)
            scatter_points.append(scatter)
            camera_names.append(img_name)
            
            # Plot camera direction
            look_dir = R[:, 2]  # Camera looks along -Z in its own coordinates
            ax.quiver(camera_pos[0], camera_pos[1], camera_pos[2],
                     look_dir[0] * arrow_length, look_dir[1] * arrow_length, look_dir[2] * arrow_length,
                     color='r', alpha=0.5)
    else:  # Legacy format
        # all transforms are w2c
        for img_name, cam_data in camera_extrinsics.items():
            R = np.array(cam_data['rotation'])
            camera_pos = np.array(cam_data['camera_pos'])
            
            if additional_transform is not None:
                R = additional_transform @ np.linalg.inv(R)
            else:
                R = np.linalg.inv(R)


    # # Plot each camera position
    # for img_name, cam_data in camera_extrinsics.items():
    #      # Get rotation matrix
    #     if additional_transform is not None:
    #         R = additional_transform @ np.linalg.inv(np.array(cam_data['rotation']))
    #     else:
    #         R = np.linalg.inv(np.array(cam_data['rotation']))

        # Get camera position
            pos = np.array(cam_data['camera_pos'])
            
            # Plot camera position as a point and store the scatter object
            scatter = ax.scatter(pos[0], pos[1], pos[2], 
                            c='blue', marker='o', picker=5, s=100)  # Increased picker radius
            scatter_points.append(scatter)
            camera_names.append(img_name)
            
            
            # Define camera direction vector (scaled for visualization)
            look_dir = R[:, 2]
            
            # Plot camera direction arrow
            ax.quiver(pos[0], pos[1], pos[2],
                    look_dir[0] * arrow_length, look_dir[1] * arrow_length, look_dir[2] * arrow_length,
                    color='r', alpha=0.5)

    # Plot central position
    center = get_central_position(camera_extrinsics)
    ax.scatter(center[0], center[1], center[2], c='red', marker='o', s=100, label=f"Center: {center}")
    ax.legend()
    
    # Set labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    
    # Set title
    plt.title('Camera poses in space')

    def on_pick(event):
        ind = event.ind[0]  # Get the index of clicked point
        for scatter, name in zip(scatter_points, camera_names):
            if event.artist == scatter:
                print(f"Camera ID: {name}")
                break

    if timestep is not None:
        def hover(event):
            nonlocal hover_ax, current_scatter
            
            if event.inaxes != ax:
                if hover_ax:
                    hover_ax.set_visible(False)
                    plt.draw()
                return
            camera_extrinsics = {}

            if "frames" in camera_extrinsics:
                for transform in camera_extrinsics["frames"]:
                    camera_extrinsics[transform["tag_id"]] = {
                        'camera_pos': np.array(transform['transform_matrix'])[:3, 3].tolist(),
                        'rotation': np.array(transform['transform_matrix'])[:3, :3].tolist()
                    }

            for scatter, name, pos in zip(scatter_points, camera_names, 
                                         [cam['camera_pos'] for cam in camera_extrinsics.values()]):
                cont, ind = scatter.contains(event)
                if cont:
                    if current_scatter == scatter:
                        return  # Already shown
                        
                    # Clean up previous hover ax
                    if hover_ax:
                        hover_ax.remove()
                        
                    try:
                        # Convert 3D position to 2D display coordinates
                        x, y, _ = proj3d.proj_transform(pos[0], pos[1], pos[2], ax.get_proj())
                        x2, y2 = ax.transData.transform((x, y))
                        xfig, yfig = fig.transFigure.inverted().transform((x2, y2))
                        
                        # Create new axes near the point
                        img_size = 0.35  # Size of image relative to figure
                        pad = 0.02  # Padding from point
                        
                        # Adjust position if near edge
                        xfig = max(0.05, min(xfig - img_size/2, 0.95 - img_size))
                        yfig = max(0.05, min(yfig + pad, 0.95 - img_size))
                        
                        hover_ax = fig.add_axes([xfig, yfig, img_size, img_size])
                        hover_ax.axis('off')
                        
                        # Load and display image
                        img = plt.imread(f'{image_dir}/{name[2:-4]}/{timestep}_img.jpg')
                        hover_im = hover_ax.imshow(img)
                        hover_ax.set_title(name, fontsize=8)
                        current_scatter = scatter
                        plt.draw()
                        return
                    except Exception as e:
                        print(f"Error loading image: {e}")
                        return
            # Hide if not hovering over point
            if hover_ax:
                hover_ax.remove()
                hover_ax = None
                plt.draw()

        fig.canvas.mpl_connect('motion_notify_event', hover)
    fig.canvas.mpl_connect('pick_event', on_pick)
    
    if output_path is not None:
        plt.savefig(output_path)
    else:
        # Show interactive plot
        plt.show()

def load_camera_extrinsics(json_path):
    """Loads camera extrinsics from a JSON file, supporting both formats:
    1. Dict with camera names as keys (legacy format).
    2. Dict with "frames" key containing list of cameras with "file_path" and "transform_matrix".
    """
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data


def get_camera_scale(path):
    with open(path, "rb") as f:
        cam_scale = pickle.load(f)
    return cam_scale


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--timestep", type=int, default=0)
    parser.add_argument("--image_dir", type=str, default="mvhumannet_format_dir/images_lr", help="Path to image directory (for hover preview)")
    parser.add_argument('--transform_coords', type=str, default=None, required=False,
                    help='Converts transforms.json output for SEVA. (Either OPENCV or OPENGL.) Expects a 3x4 numpy array txt file.')
    parser.add_argument('--transforms_path', type=str, default="mvhumannet_format_dir/transforms.json", help="Path to transforms.json file")
    args = parser.parse_args()
    camera_scale_path = os.path.join(os.path.dirname(args.transforms_path), "camera_scale.pkl")
    timestep = f"{str(args.timestep).zfill(4)}"
    cam_ex = load_camera_extrinsics(args.transforms_path)

    try:
        cam_scale = get_camera_scale(camera_scale_path)
        print(f"Using camera scale: {cam_scale}")
    except FileNotFoundError:
        print(f"Camera scale file not found at {camera_scale_path}. Using default scale of 1.0. This is fine if transforms have already been scaled.")
        cam_scale = 1.0
    
    cam_ex_scaled = cam_ex

    # apply camera scale
    if "frames" in cam_ex:
        for frame in cam_ex["frames"]:
            for i in range(3):
                frame["transform_matrix"][i][3] *= cam_scale
    else:  # legacy format
        for k, v in cam_ex.items():
            cam_ex_scaled[k]["camera_pos"] = [t * cam_scale for t in cam_ex[k]["camera_pos"]]

    if args.transform_coords is not None:
        transform_coords = np.loadtxt(args.transform_coords)
        transform = transform_coords[:3,:3]
    else: 
        transform = None

    arrow_length = 200
    if args.timestep > 0: # if connected to a timestep
        visualize_camera_poses(cam_ex_scaled, 
        additional_transform=transform, 
        timestep=timestep, 
        image_dir="mvhumannet_format_dir/images_lr",
        arrow_length=arrow_length)
    else:
        visualize_camera_poses(cam_ex_scaled, 
        additional_transform=transform,
        arrow_length=arrow_length)

# notes
# Z is height (from this transform)
# X and Y is the ground plane