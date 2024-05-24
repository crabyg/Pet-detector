
import detect_car as car
import org_to_CLAHE as CLAHE
import cv2
import matplotlib.pyplot as plt
from clip_function import *
def main():
    crop_img = car.run(
        # weights= 'C:/Users/MMC/Desktop/Pet-detector/yolov5m_pet.pt',  # 새로운 가중치 파일 경로
        source='C:/Users/MMC/CLIP/pet-detector/car_data/sample1.jpg',  # 새로운 이미지 소스 경로
        # imgsz=(800, 800),  # 변경된 이미지 크기
        # conf_thres=0.5  # 변경된 신뢰도 임계값
    )
    
    crop_img_clahe = CLAHE.apply_CLAHE(crop_img) # b =[top right clahe]

    # cv2.imwrite('runs/crop.jpg', crop_img)
    # cv2.imwrite('runs/crop_clahe.jpg', crop_img_clahe)
    CLIP_function(crop_img_clahe)

if __name__ == "__main__":
    main()
