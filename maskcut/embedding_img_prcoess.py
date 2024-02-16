import os
from PIL import Image

# Assuming your base directory where the folders are located is 'base_directory'
# base_directory = "/home/jungseok/data/zpool_dataset/2024-01-21-19-25-39_out/"
# base_directory2 = "/home/jungseok/data/zpool_dataset/stitched_images/"
# similarity_file = os.path.join(base_directory, 'similarity_rankings.txt')

base_directory = "/home/jungseok/data/kelpie/kelpie_filtered_out/"
base_directory1 = "/home/jungseok/data/kelpie/"
base_directory2 = "/home/jungseok/data/kelpie/kelpie_stitched_images_mean/"
similarity_file = os.path.join(base_directory1, 'similarity_rankings.txt')

# def stitch_images(img1_path, img2_path, output_filename):
#     """Stitch two images side by side and save the result."""
#     image1 = Image.open(img1_path)
#     image2 = Image.open(img2_path)
    
#     # Create a new image with a width equal to the sum of both images' widths
#     # and a height equal to the maximum height of the two images
#     new_image = Image.new('RGB', (image1.width + image2.width, max(image1.height, image2.height)))
    
#     # Paste the two images into this new image
#     new_image.paste(image1, (0, 0))
#     new_image.paste(image2, (image1.width, 0))
    
#     # Save the new image
#     new_image.save(output_filename)

def stitch_images(img1_path, img2_path, output_filename, scale_factor=0.3):
    """Stitch two images side by side, resize the result, and save it."""
    image1 = Image.open(img1_path)
    image2 = Image.open(img2_path)
    
    # Create a new image with a width equal to the sum of both images' widths
    # and a height equal to the maximum height of the two images
    new_image_width = image1.width + image2.width
    new_image_height = max(image1.height, image2.height)
    new_image = Image.new('RGB', (new_image_width, new_image_height))
    
    # Paste the two images into this new image
    new_image.paste(image1, (0, 0))
    new_image.paste(image2, (image1.width, 0))
    
    # Resize the new image based on the scale factor
    # The scale_factor determines the final size; for example, 0.5 will reduce the size to 50%
    resized_image = new_image.resize((int(new_image_width * scale_factor), int(new_image_height * scale_factor)))
    
    # Save the resized new image
    resized_image.save(output_filename)

def process_similarity_file(similarity_file, base_directory):
    with open(similarity_file, 'r') as file:
        for line in file:
            # Extract filename1 and filename2 from each line
            parts = line.strip().split(': ')[0].split(' - ')
            # print(line.strip().split(': ')[-1])
            tensor_value_part = line.strip().split(': ')[-1]
            similarity = tensor_value_part.replace("tensor([", "").replace("])", "").strip()

            if len(parts) < 2:
                continue  # Skip lines that don't match the expected format
            filename1, filename2 = parts
            filename1 = filename1.strip().split(' ')[1]
            extracted1 = '_'.join(filename1.split('_', 2)[:2])
            extracted2 = '_'.join(filename2.split('_', 2)[:2])
            print(extracted1, extracted2)

            # Construct the path to the image files
            img1_path = os.path.join(base_directory, extracted1, f"{extracted1}_seg.jpg")
            img2_path = os.path.join(base_directory, extracted2, f"{extracted2}_seg.jpg")
            
            # Check if both image files exist
            if os.path.exists(img1_path) and os.path.exists(img2_path):
                # Stitch the images together
                # output_filename = f"{extracted1}_{extracted2}_similarity.jpg"
                output_filename = os.path.join(base_directory2, f"{similarity}_{extracted1}_{extracted2}.jpg")
                stitch_images(img1_path, img2_path, output_filename)
                print(f"Stitched image saved as: {output_filename}")
            else:
                print(f"One or both image files not found for: {filename1} and {filename2}")

# Call the function to process the similarity file and stitch images
process_similarity_file(similarity_file, base_directory)
