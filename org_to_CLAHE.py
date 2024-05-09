import cv2
import albumentations as A

def apply_CLAHE(original_image):
    augment = A.Compose([
        A.Resize(width=640,height=640),
        A.CLAHE(always_apply=True, p=1.0, clip_limit=(1,4), tile_grid_size=(8, 8)),
    ])
    
    augmented_image = augment(image=original_image)['image']
    
    return augmented_image