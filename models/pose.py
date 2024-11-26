import os
import torch
from transformers import ViTImageProcessor, ViTForImageClassification
from init_model import InitModel

class PoseModel(InitModel):
    """자세 분류 모델"""
    def __init__(self):
        self.real_label = set([0,1,2,3,4,5])
        self.label_number = 6
        self.label_name = []
        
    def predict(self, input_data):        
        """
        모델 예측 메서드
        input_data: image object
        """
        self.init_cls_model("ViewClassification", self.real_label, self.label_number) # 분류 model 초기화
        inputs = self.processor(images=input_data, return_tensors="pt")
        
        with torch.no_grad():  
            outputs = self.classification_model(**inputs)
            
        logits = outputs.logits
        predicted_class = torch.argmax(logits, dim=-1)
        
        return predicted_class.item()