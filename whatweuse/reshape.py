import cv2
import glob
    
image_path = './test3/'

iphone_resacle = (1008,756)
samsung_resacel = (1008, 1008)

# load the images
image_files = glob.glob(image_path + "*.jpg")
images = []
for file in image_files:
    print(file)
    image = cv2.imread(file)
    print(image.shape)
    image = cv2.resize(image, (1008, 1008))
    cv2.imwrite(file, image)
    