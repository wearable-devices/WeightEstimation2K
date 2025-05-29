import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os

import plotly.graph_objects as go
from PIL import Image
import io
import  tensorflow as tf

def generate_tensor_colors(num_tensors):
    """
    Generate distinct colors for tensor visualization.

    Parameters:
    num_tensors (int): Number of tensors to generate colors for

    Returns:
    list: List of RGB color strings compatible with Plotly
    """
    # Define custom color palettes with good perceptual separation
    color_palettes = {
        'distinct': [
            (230, 25, 75), (60, 180, 75), (255, 225, 25), (0, 130, 200), (245, 130, 48),
            (145, 30, 180), (70, 240, 240), (240, 50, 230), (210, 245, 60), (250, 190, 212),
            (0, 128, 128), (220, 190, 255), (170, 110, 40), (255, 250, 200), (128, 0, 0),
            (170, 255, 195), (128, 128, 0), (255, 215, 180), (0, 0, 128), (128, 128, 128)
        ],
        'tableau20': [
            (31, 119, 180), (255, 127, 14), (44, 160, 44), (214, 39, 40), (148, 103, 189),
            (140, 86, 75), (227, 119, 194), (127, 127, 127), (188, 189, 34), (23, 190, 207),
            (174, 199, 232), (255, 187, 120), (152, 223, 138), (255, 152, 150), (197, 176, 213),
            (196, 156, 148), (247, 182, 210), (199, 199, 199), (219, 219, 141), (158, 218, 229)
        ],
        'plotly_dark': [
            (99, 110, 250), (239, 85, 59), (0, 204, 150), (171, 99, 250), (255, 161, 90),
            (25, 211, 243), (255, 102, 146), (182, 232, 128), (255, 151, 255), (254, 203, 82)
        ]
    }

    # Choose the appropriate palette based on number of tensors
    if num_tensors <= 10:
        palette = color_palettes['distinct'][:10]  # Best for small number of distinct items
    else:
        palette = color_palettes['tableau20']  # More colors for larger sets

    colors = []

    # If we need more colors than are in the palette
    if num_tensors <= len(palette):
        colors = palette[:num_tensors]
    else:
        # Use all colors from the palette
        colors = palette.copy()

        # Then repeat colors with variation if needed
        import random
        for i in range(len(palette), num_tensors):
            base_idx = i % len(palette)
            variation = 30  # Add some variation to repeated colors
            r, g, b = palette[base_idx]

            # Modify the color slightly for variation
            r = max(0, min(255, r + random.randint(-variation // 2, variation // 2)))
            g = max(0, min(255, g + random.randint(-variation // 2, variation // 2)))
            b = max(0, min(255, b + random.randint(-variation // 2, variation // 2)))

            colors.append((r, g, b))

    # Convert to the format expected by Plotly
    return [f'rgb({r}, {g}, {b})' for r, g, b in colors]




def plot_sequences(list_of_data):
    plt.figure(figsize=(12, 6))

    # Create x-axis indices for each sequence
    # lengths = [np.arange(len(data)) for data in list_of_data]
    xx = [np.arange(len(data)) for data in list_of_data]
    # x1 = np.arange(len(snc1))
    # x2 = np.arange(len(snc2))
    # x3 = np.arange(len(snc3))

    # Plot each sequence with different colors and markers
    colors = ['b.-', 'r.-', 'g.-']
    for i,x in enumerate(xx):
        plt.plot(x, list_of_data[i], colors[i], label=f'SNC{i+1}', markersize=8)
    # plt.plot(x1, snc1, 'b.-', label='SNC1', markersize=8)
    # plt.plot(x2, snc2, 'r.-', label='SNC2', markersize=8)
    # plt.plot(x3, snc3, 'g.-', label='SNC3', markersize=8)

    # Customize the plot
    plt.title('SNC Sequences Comparison')
    plt.xlabel('Sample Index')
    plt.ylabel('Value')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()

    # Adjust layout to prevent label clipping
    plt.tight_layout()

    # Show the plot
    plt.show()


def plot_heatmap_tensors(tensors, names=None, title=None,normalize=False, save_path=None):
    # Create a figure with two subplots
    # fig, axes = plt.subplots(2, (len(tensors)+1)//2, figsize=(20, 8))
    n = len(tensors)
    fig, axes = plt.subplots( n,1, figsize=(20, 8))
    #Adjust the layout to make room for the title
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    if title is not None:
        fig.canvas.manager.set_window_title(title)

    # If there's only one tensor, make axes into an array
    if n == 1:
        axes = np.array([axes])

    # Plot heatmap for tensor1
    for i, tensor in enumerate(tensors):
        if len(tensor.shape) == 1:
            # axes[i].set_ylim(-0.5, 0.5)
            axes[i].plot(tensor, linewidth=2.5)
        else:
            tensor_to_plot = np.log(tensor - np.min(tensor)) / (np.max(tensor) - np.min(tensor)) if normalize else tensor

            im = axes[i].imshow(tensor_to_plot, cmap='viridis', aspect='auto')
        if names is not None:
            axes[i].set_title(names[i])
            # plt.colorbar(im, ax=axes[i])

    # Adjust layout and display the plot
    plt.tight_layout()
    plt.show()


def plot_movie_tensors(tensors, names=None, title=None, normalize=False, save_path=None,im_min =None, im_max=None, fps=10):
    """
    Create a movie from a list of tensors where the first dimension is time/frame number.

    Args:
        tensors: List of tensors, each with shape [frames, ...other dimensions]
        names: Optional list of names for each tensor plot
        title: Title for the animation
        normalize: Whether to normalize the data
        save_path: Path to save the animation
        fps: Frames per second for the animation
    """
    # Get number of frames from the first tensor
    n_frames = tensors[0].shape[0]
    n = len(tensors)

    # Create figure and axes
    fig, axes = plt.subplots(n, 1, figsize=(20, 8))
    if n == 1:
        axes = np.array([axes])

    if title is not None:
        fig.canvas.manager.set_window_title(title)

    # Initialize plots
    plots = []
    for i, tensor in enumerate(tensors):
        if len(tensor.shape) == 2:  # 1D data over time
            line, = axes[i].plot([], [], lw=2)
            axes[i].set_xlim(0, tensor.shape[1])
            axes[i].set_ylim(np.min(tensor), np.max(tensor))
            plots.append(line)
        else:  # 2D data over time
            if normalize:
                vmin = np.min(tensor)
                vmax = np.max(tensor)
                normalized_tensor = (tensor - vmin) / (vmax - vmin)
                if np.any(tensor < 0):  # Center around zero if there are negative values
                    normalized_tensor = 2 * normalized_tensor - 1
                tensor = normalized_tensor

            vmin = im_min if im_min is not None else np.min(tensor)
            vmax = im_max if im_max is not None else np.max(tensor)
            im = axes[i].imshow(tensor[0], cmap='viridis', aspect='auto',
                                vmin=vmin, vmax=vmax)
            # fig.colorbar(im, ax=axes[i])
            plots.append(im)

        if names is not None:
            axes[i].set_title(names[i])

    # Animation update function
    def update(frame):
        for i, tensor in enumerate(tensors):
            if len(tensor.shape) == 2:  # 1D data
                plots[i].set_data(np.arange(tensor.shape[1]), tensor[frame])
            else:  # 2D data
                plots[i].set_array(tensor[frame])
        return plots

    # Create animation
    ani = animation.FuncAnimation(
        fig, update, frames=n_frames,
        interval=1000 / fps,  # interval in milliseconds
        blit=True
    )

    plt.tight_layout()

    # Save if path is provided
    if save_path is not None:
        ani.save(save_path, writer='pillow', fps=fps)

    plt.show()

def snc_scat_all(snc1_data,snc2_data, snc3_data, scattered_snc1, scattered_snc2, scattered_snc3,
                 scattered_snc11, scattered_snc22, scattered_snc33):
    plt.subplot(3, 3, 1)
    plt.ylim([-0.3, 0.3])
    plt.margins(0)
    plt.plot(snc1_data)
    plt.plot(snc1_data, '.b',  markersize=2)

    plt.subplot(3, 3, 2)
    plt.ylim([-0.3, 0.3])
    plt.margins(0)
    plt.plot(snc2_data)
    plt.plot(snc2_data, '.b',  markersize=2)

    plt.subplot(3, 3, 3)
    plt.ylim([-0.3, 0.3])
    plt.margins(0)
    plt.plot(snc3_data)
    plt.plot(snc3_data, '.b', markersize=2)

    plt.subplot(3, 3, 4)
    plt.imshow(scattered_snc1[0], cmap='viridis', aspect='auto', vmax=0.045)

    plt.subplot(3, 3, 5)
    plt.imshow(scattered_snc2[0], cmap='viridis', aspect='auto', vmax=0.045)

    plt.subplot(3, 3, 6)
    plt.imshow(scattered_snc3[0], cmap='viridis', aspect='auto', vmax=0.045)

    plt.subplot(3, 3, 7)
    plt.imshow(scattered_snc11[0], cmap='viridis', aspect='auto', vmax=0.045)

    plt.subplot(3, 3, 8)
    plt.imshow(scattered_snc22[0], cmap='viridis', aspect='auto', vmax=0.045)

    plt.subplot(3, 3, 9)
    plt.imshow(scattered_snc33[0], cmap='viridis', aspect='auto', vmax=0.045)

    plt.show()

def plot_feature_space(persons_dict, num_of_components=2, save_path=None, img_name=''):
    # Color map for labels
    color_map = {
        0: 'red',
        0.5: 'pink',
        1: 'blue',
        2: 'green',
        4: 'yellow',
        6: 'brown',
        8: 'black'
    }

    color_map_2 = {
        0: 'crimson',
        0.5: 'pink',
        1: 'navy',
        2: 'forestgreen',  # Changed from 'forest_green' to 'forestgreen'
        4: 'gold',
        6: 'sienna',
        8: 'dimgray'  # Changed from 'charcoal' to 'dimgray' as charcoal isn't a standard CSS color
    }

    # Marker map for persons
    markers = ['square' , 'square-open', 'circle','square', 'diamond', 'cross', 'x', 'diamond-open']
    marker_map = {person: markers[i % len(markers)] for i, person in enumerate(persons_dict.keys())}

    # Create the plot
    fig = go.Figure()

    user_num = 0
    for person, labels in persons_dict.items():
        user_num += 1
        used_color_map = color_map_2 if user_num % 2 == 0 else color_map
        for label, tensor in labels.items():
            if num_of_components == 1:
                # Set all y-values to a constant (e.g., 0) to place them on one line
                fig.add_trace(go.Scatter(
                    x=tensor[:, 0],  # Use the single component for x-axis
                    y=[0] * len(tensor),  # Create array of zeros same length as tensor
                    mode='markers',
                    marker=dict(
                        size=8,
                        color=used_color_map[label],
                        symbol=marker_map[person],
                        opacity=0.8
                    ),
                    name=f"{person} - {label}"
                ))
            elif num_of_components == 2:
                fig.add_trace(go.Scatter(
                    x=tensor[:, 0],
                    y=tensor[:, 1],
                    mode='markers',
                    marker=dict(
                        size=8,
                        color=used_color_map[label],
                        symbol=marker_map[person],
                        opacity=0.8
                    ),
                    name=f"{person} - {label}"
                ))
            else:
                fig.add_trace(go.Scatter3d(
                    x=tensor[:, 0],
                    y=tensor[:, 1],
                    z=tensor[:, 2],
                    mode='markers',
                    marker=dict(
                        size=8,
                        color=used_color_map[label],
                        symbol=marker_map[person],
                        opacity=0.8
                    ),
                    # name=f"{person} - {label}"
                    name=f" {person} - {label} g"
                ))

    # Update layout
    fig.update_layout(
        title=f"Feature Space Visualization",  # of {self.layer_name}_proj_{self.proj}_name_{self.picture_name}",
        xaxis_title="X",
        yaxis_title="Y",
        legend_title="User - Weight"
    )

    fig.show()
    if save_path is not None:
        # Save the plot as HTML
        html_path = os.path.join(save_path,
                                 f"{img_name}.html")
        fig.write_html(html_path, full_html=True)

    # Create frames for animation
    # frames = []
    # for i in range(0, 360, 10):  # Rotate 360 degrees in steps of 10
    #     fig.update_layout(scene_camera=dict(eye=dict(x=1.25 * np.cos(np.radians(i)),
    #                                                  y=1.25 * np.sin(np.radians(i)),
    #                                                  z=0.5)))
    #     img_bytes = fig.to_image(format="png")
    #     img = Image.open(io.BytesIO(img_bytes))
    #     frames.append(img)
    #
    # # Save frames as GIF
    # gif_path = os.path.join(self.trial_dir,
    #                         f"{self.layer_name}_{self.proj}_{self.metric}_{self.num_of_components}_comp_feature_space_{self.picture_name}.gif")
    #
    # frames[0].save(gif_path, save_all=True, append_images=frames[1:], duration=200, loop=0)
    #
    # print(f"Animated GIF saved to: {gif_path}")

    # print(f"Interactive plot saved to: {html_path}")

def plot_curves(dict, show=False, save_path=None, picture_name='_'):
    # Plot both training and validation losses
    plt.figure(figsize=(12, 5))
    for name, value in dict.items():
        plt.plot(value, label=name)

    # plt.title(f'Adaptation Progress for {user}')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.legend()
    if show:
        plt.show()
    if save_path is not None:
        plt.savefig(os.path.join(save_path, picture_name))
    # Clear the current figure
    plt.close()

def interactive_plot_tensors_3d_improved(tensor_dict, output_html=None, title='Interactive 3D Visualization of Tensors'):
    """
    Create an interactive 3D plot of tensors with the ability to toggle visibility
    by clicking on tensor names in the legend.

    Parameters:
    tensor_dict (dict): Dictionary of {tensor_name: tensor} where each tensor has shape (n, 3)
    output_html (str, optional): Path to save the HTML file. If None, the plot is displayed inline.

    Returns:
    plotly.graph_objects.Figure: The interactive figure object
    """
    # Create a new figure
    fig = go.Figure()

    # Get colors for the tensors using our improved color generation function
    num_tensors = len(tensor_dict)
    colors = generate_tensor_colors(num_tensors)

    # Add each tensor as a separate scatter3d trace
    for i, (tensor_name, tensor) in enumerate(tensor_dict.items()):
        # Convert tensor to numpy array if it's a TF tensor
        if isinstance(tensor, tf.Tensor):
            points = tensor.numpy()
        else:
            points = tensor

        # Ensure shape is correct
        if points.shape[1] != 3:
            raise ValueError(f"Tensor {tensor_name} must have shape (n, 3), got {points.shape}")

        # Extract x, y, z coordinates
        x, y, z = points[:, 0], points[:, 1], points[:, 2]

        # Add the scatter trace
        fig.add_trace(go.Scatter3d(
            x=x, y=y, z=z,
            mode='markers',
            marker=dict(
                size=4,
                color=colors[i],
                opacity=0.7
            ),
            name=tensor_name
        ))

    # Update layout for better visualization
    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z'
        ),
        legend=dict(
            title="Tensors (click to toggle visibility)",
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        ),
        margin=dict(l=0, r=0, b=0, t=30)
    )

    # Save to HTML file if specified
    if output_html:
        fig.write_html(output_html)
        print(f"Interactive plot saved to {output_html}")

    return fig