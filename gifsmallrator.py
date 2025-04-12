from PIL import Image, ImageSequence
import os

def reduce_gif_size_pillow_resize(input_path, output_path, target_mb=10, scale_factor=0.8):
    """
    Reduces GIF size primarily by resizing using Pillow.

    Args:
        input_path (str): Path to the input GIF file.
        output_path (str): Path to save the resized GIF file.
        target_mb (int): The target maximum file size in megabytes.
                         (Note: Pillow doesn't optimize like gifsicle,
                          so hitting an exact target by resizing alone is hard).
        scale_factor (float): Factor to scale dimensions by (e.g., 0.8 = 80%).

    Returns:
        bool: True if resizing was performed, False on error.
              Doesn't guarantee target size is met.
    """
    if not os.path.exists(input_path):
        print(f"ERROR: Input file not found: {input_path}")
        return False

    try:
        img = Image.open(input_path)

        # Get original properties
        original_duration = img.info.get('duration', 100) # Default frame duration
        original_loop = img.info.get('loop', 0) # Default loop count (0 = infinite)

        frames = []
        new_width = int(img.width * scale_factor)
        new_height = int(img.height * scale_factor)
        new_size = (new_width, new_height)

        print(f"Original size: {img.width}x{img.height}")
        print(f"Resizing to: {new_width}x{new_height} (scale factor: {scale_factor})")

        for frame in ImageSequence.Iterator(img):
            # Ensure frame is RGBA or RGB before resizing for better quality
            # and convert back to 'P' (palettized) for GIF saving
            frame = frame.convert('RGBA')
            resized_frame = frame.resize(new_size, Image.Resampling.LANCZOS)
            # Convert back to Palettized ('P') mode for GIF.
            # Using ADAPTIVE palette for each frame - might not be optimal for size.
            # A global palette is complex to generate well with Pillow alone.
            final_frame = resized_frame.convert('P', palette=Image.Palette.ADAPTIVE)
            frames.append(final_frame)

        if not frames:
            print("ERROR: No frames extracted from GIF.")
            return False

        # Save the resized frames as a new GIF
        frames[0].save(
            output_path,
            save_all=True,
            append_images=frames[1:],
            duration=original_duration,
            loop=original_loop,
            optimize=True # Basic Pillow optimization
        )

        initial_size_mb = os.path.getsize(input_path) / (1024 * 1024)
        output_size_mb = os.path.getsize(output_path) / (1024 * 1024)

        print(f"Original size: {initial_size_mb:.2f} MB")
        print(f"Resized output size: {output_size_mb:.2f} MB")

        if output_size_mb <= target_mb:
             print(f"Target size of {target_mb} MB met or bettered.")
        else:
             print(f"Target size of {target_mb} MB NOT met. Consider a smaller scale_factor or using gifsicle.")

        return True

    except FileNotFoundError:
         print(f"ERROR: Input file not found: {input_path}")
         return False
    except Exception as e:
        print(f"An error occurred during Pillow processing: {e}")
        return False

# --- Usage Example ---
input_gif = "schaduw_animatie_20250721.gif"  # Replace with your input file path
output_gif = "schaduw_animatie_20250721_pillow_resized.gif" # Replace with desired output file path

# Try resizing to 80% - adjust scale_factor as needed
reduce_gif_size_pillow_resize(input_gif, output_gif, target_mb=10, scale_factor=0.8)