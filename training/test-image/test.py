import cv2
from tkinter import Tk
from tkinter.filedialog import askopenfilename
from skimage import feature, data, exposure
from skimage.color import rgb2gray
from skimage.feature import hog

def process_image(image_path):
    # Read the image
    image = cv2.imread(image_path)
    if image is None:
        print("Error: Unable to load image.")
        return None
    
    # image = data.coffee()
    
    # Resize the image to 224 x 224
    processed_image = cv2.resize(image, (224, 224))
    
    processed_image = rgb2gray(processed_image)
    
    # Enhance contrast using histogram equalization
    processed_image = cv2.equalizeHist((processed_image * 255).astype('uint8'))

    fd, hog_image = hog(processed_image, orientations=8, pixels_per_cell=(8, 8), cells_per_block=(1, 1), visualize=True)

    hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))

    return hog_image_rescaled

def main():
    Tk().withdraw()
    # image_path = askopenfilename(title="Select an Image", filetypes=[("Image Files", "*.jpg;*.png;*.jpeg")])
    image_path = "C:/Users/emilia/Downloads/WildLens(1)/test.png"
    
    if not image_path:
        print("No file selected.")
        return

    processed_image = process_image(image_path)
    if processed_image is None:
        return

    cv2.imshow("Processed Image", processed_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()