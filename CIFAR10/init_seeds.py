# -*- coding: utf-8 -*-
import os
import numpy as np
from keras.datasets import cifar10
from keras.preprocessing.image import array_to_img

# CIFAR-10 데이터셋 로드
(_, _), (x_test, y_test) = cifar10.load_data()

# ./seeds/ 디렉터리 생성
output_dir = 'seeds/'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 클래스별로 200개씩 저장
num_per_class = 200
for class_id in range(10):
    class_indices = np.where(y_test == class_id)[0]  # 해당 클래스의 인덱스 선택
    selected_indices = np.random.permutation(class_indices)[:num_per_class]  # 각 클래스에서 200개 무작위로 선택

    # 이미지 저장
    for i, idx in enumerate(selected_indices):
        img = array_to_img(x_test[idx])  # CIFAR-10 이미지를 이미지 파일로 변환
        img.save(os.path.join(output_dir, 'class_{}_image_{}.png'.format(class_id, i)))

print('Each of 10 classes saved {} images to {}, totaling {} images.'.format(num_per_class, output_dir, num_per_class * 10))

