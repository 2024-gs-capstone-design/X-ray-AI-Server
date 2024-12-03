from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
from PIL import Image
import io
import base64
import sys, os
sys.path.append(os.path.join(os.getcwd(), 'models'))
import models
from pydantic import BaseModel

class ImagePath(BaseModel):
    path: str
    
# FastAPI 인스턴스 생성
app = FastAPI()

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 가상의 모델 함수 (Object Detection 결과 반환)
def model_image(image: Image.Image) -> Image.Image:
    # Object Detection 로직을 여기에 구현
    # 이 예제에서는 입력 이미지를 그대로 반환 (테스트용)
    # 2개의 모델 객체 초기화
    stage1_model = models.PoseModel()
    stage2_model = models.DetailedPoseModel()

    # 자세 탐지 모델
    output1 = stage1_model.predict(image)
    # 공통 + 세부 자세 탐지 모델
    forResult = stage2_model._get_class_name(output1)
    _, predicted_img = stage2_model.predict(output1, image) # output2는 output과 예측 이미지를 내놓는다.
    
    # self.right_text_label.append(f"자세: {forResult[0]}")
    
    return predicted_img  # 추후 이 부분에 BBox 그린 결과 반환

# @app.post("/file/")
# async def upload_file(file: UploadFile = File(...)):
#     # 업로드된 파일을 메모리에서 읽음
#     content = await file.read()
#     original_image = Image.open(io.BytesIO(content))

#     # 모델 함수 호출
#     processed_image = model_image(original_image)

#     # 결과 이미지를 Base64로 변환
#     buffered = io.BytesIO()
#     processed_image.save(buffered, format="PNG")
#     buffered.seek(0)
#     base64_image = base64.b64encode(buffered.getvalue()).decode("utf-8")

#     # JSONResponse로 결과 반환
#     return JSONResponse(content={
#         "processed_image": base64_image  # Base64로 인코딩된 모델 처리 결과 이미지
#     })
@app.post("/file/")
async def process_image(image_path: ImagePath):
    # 이미지 경로 읽기
    input_path = image_path.path

    if not os.path.exists(input_path):
        return {"error": "Image path does not exist"}

    # 이미지 열기
    original_image = Image.open(input_path)

    # 모델 함수 호출
    processed_image = model_image(original_image)

    # 저장 경로 생성
    base_name, ext = os.path.splitext(os.path.basename(input_path))
    output_path = os.path.join(
        os.path.dirname(input_path), f"{base_name}_predict.png"
    )

    # 처리된 이미지 저장
    processed_image.save(output_path, format="PNG")

    # 저장 경로 반환
    return {"processed_image_path": output_path}

if __name__ == "__main__":
    uvicorn.run("main:app", host = "127.0.0.1",port=8000,
            reload = True)