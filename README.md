# WidePredict

**2021 Konkuk Univ. Electric &amp; Electronic Engineering Capstone Design**

![image](https://github.com/user-attachments/assets/567017ba-43ff-4302-ae8b-ed5ed11242e0)

---

## **프로젝트 개요**

- **TensorRT 최적화 YOLO 모델**과 **SGAN(Social GAN)**을 활용하여 객체를 탐지하고, 추적하며, 이동 경로를 예측합니다.  
- 두 대의 스테레오 카메라를 통해 넓은 파노라마 영상으로 합성, 2D의 픽셀 좌표계를 3D의 월드 좌표계로 변환합니다.  
- 변환된 3D 공간에서 객체 위치를 추정하고, 이를 기반으로 경로를 예측하여 실시간으로 시각화합니다.

---

## **주요 기능**

1. **실시간 객체 탐지**

   - TensorRT를 이용하여 YOLO 모델 최적화.
   - 탐지된 객체의 위치와 클래스 표시.
   - 카메라 행렬 및 변환 벡터를 활용해 픽셀 좌표를 월드 좌표로 변환.

   ![image](https://github.com/user-attachments/assets/e36cbc6c-bf3e-4f3c-aa20-298835aefc23)

2. **객체 추적 및 경로 예측**

   - Sort 알고리즘을 사용한 객체 추적.
   - SGAN을 이용하여 사회적 상호작용을 고려한 이동 경로 예측.

   ![image](https://github.com/user-attachments/assets/2148c3dd-7968-4260-a3cc-7a232c8fcc3b)

3. **스테레오 카메라 기반 3D 좌표 추정**

   - 두 대의 카메라에서 입력된 영상을 ORB(Oriented FAST and Rotated BRIEF)로 매칭하여 병합.

   ![image](https://github.com/user-attachments/assets/0e48a54b-80ba-4574-b621-4af912196cdb)
   ![image](https://github.com/user-attachments/assets/c12af739-6986-4fb1-aec8-ba8469e37dc9)

4. **속도 제어 및 경고 시각화**

   - 객체가 특정 영역에 접근하면 속도 감소를 시뮬레이션.
   - 예측 경로와 경계선을 통해 위험 구역 표시.

   ![image](https://github.com/user-attachments/assets/5b9edafc-d2b3-4bf3-8d29-cbc3666a2100)
   ![image](https://github.com/user-attachments/assets/bbbd7b9e-ce41-48a2-9372-34b090322f40)

---

## **실행 환경**

- **하드웨어:**

  - Jetson Nano
  - 2대의 좌, 우 스테레오 카메라

- **소프트웨어:**
  - numpy==1.18.5
  - torch==1.2.0
  - torchvision==0.4.0
  - opencv-python==4.1.1.26
  - attrdict==2.0.0
  - matplotlib==3.3.4
  - filterpy==1.4.5

---

## **모델 파일 준비**

#### (1) YOLO 모델 (`yolo/[모델 이름].trt`)

- TensorRT로 변환된 YOLO 모델 파일이 필요합니다.
- 아래 링크를 통해 YOLO 모델 파일을 다운로드한 후, `yolo/` 디렉토리에 저장하세요:
  - [YOLO 모델 다운로드 링크](https://example.com/yolo-model)

#### (2) SGAN 모델 (`sgan/scripts/models/[모델 이름].pt`)

- SGAN 경로 예측을 위해 사전 학습된 모델 파일이 필요합니다.
- 아래 링크에서 SGAN 모델 파일을 다운로드한 후, `sgan/scripts/models/` 디렉토리에 저장하세요:
  - [SGAN 모델 다운로드 링크](https://example.com/sgan-model)

---

#### **파일 구조 예시**

```plaintext
WidePredict/
├── pedestrian_predict_panorama.py           # 메인 실행 스크립트
├── sgan/
│   └── scripts/
│       └── models/
│           └── zara2_12_model.pt  # SGAN 모델
├── yolo/
│   └── yolov4-416.trt    # YOLO 모델 (TensorRT)
├── convert_coord/
│   ├── Original camera matrix.npy
│   ├── RVec.npy
│   ├── TVec.npy
├── requirements.txt      # Python 패키지 의존성
└── README.md             # 프로젝트 설명 파일
```

---

## **참고 자료**

- [TensorRT YOLO GitHub 저장소](https://github.com/NVIDIA-AI-IOT/tensorrt_demos): TensorRT를 사용한 YOLO 모델 최적화 및 변환 가이드.
- [SGAN GitHub 저장소](https://github.com/agrimgupta92/sgan): SGAN(Social GAN) 모델 코드 및 사전 학습된 모델 제공.
- [YOLO 모델 정보](https://pjreddie.com/darknet/yolo/): YOLO(You Only Look Once) 모델 설명 및 다운로드.

---

## **데모 영상**

<sup>[WidePredict](https://youtu.be/uzITWY4wk8c)</sup>

## **문의**

질문이나 제안 사항이 있다면 아래로 연락 주세요  
이메일: columnwise99@gmail.com  
GitHub Issues를 통해서도 문의 가능합니다.
