import cv2
import numpy as np
import os

def generate_aruco_markers(dictionary_id, marker_ids, marker_size, output_folder, margin=10):
    """
    Generate Aruco markers and save them in the specified output folder.

    Parameters:
    - dictionary_id: ID of the Aruco dictionary to use (e.g., cv2.aruco.DICT_6X6_250)
    - marker_ids: List of marker IDs to generate
    - marker_size: Size of the marker in pixels
    - output_folder: Folder to save the generated markers
    - margin: Margin size in pixels to add around the marker
    """
    # Create the output folder if it does not exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Load the specified Aruco dictionary
    aruco_dict = cv2.aruco.Dictionary_get(dictionary_id)

    for marker_id in marker_ids:
        # Generate the marker image
        marker_image = cv2.aruco.drawMarker(aruco_dict, marker_id, marker_size)

        # Create a larger image with a white background
        marker_with_margin = np.ones((marker_size + 2 * margin, marker_size + 2 * margin), dtype=np.uint8) * 255

        # Place the marker image in the center of the larger image
        marker_with_margin[margin:margin + marker_size, margin:margin + marker_size] = marker_image

        # Save the marker image with margin
        output_path = os.path.join(output_folder, f"aruco_marker_{marker_id}.png")
        cv2.imwrite(output_path, marker_with_margin)
        print(f"Marker {marker_id} saved as {output_path}")

if __name__ == '__main__':
    # Example usage
    dictionary_id = cv2.aruco.DICT_6X6_250
    marker_ids = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]  # List of marker IDs to generate
    marker_size = 200  # Size of the marker in pixels
    output_folder = '/home/bal/overlay_noetic_ws/src/active_inference_planner/data/aruco_markers'  # Folder to save the generated markers

    generate_aruco_markers(dictionary_id, marker_ids, marker_size, output_folder, margin=10)
