import random

# พจนานุกรมคำพ้องความหมายทางกฎหมาย
LEGAL_SYNONYMS = {
    "ละเมิด": ["ฝ่าฝืน", "กระทำผิด", "ล่วงสิทธิ"],
    "จำหน่าย": ["ขาย", "เผยแพร่", "กระจายสินค้า"],
    "ปลอมแปลง": ["ทำเทียม", "เลียนแบบ"]
}

def augment_legal_text(text):
    words = text.split()
    new_words = words.copy()
    
    for i, word in enumerate(words):
        if word in LEGAL_SYNONYMS:
            new_words[i] = random.choice(LEGAL_SYNONYMS[word])
            
    return " ".join(new_words)

# ทดสอบ Data Augmentation
original = "จำเลย ละเมิด และ จำหน่าย สินค้า"
augmented = augment_legal_text(original)

print(f"--- 1. Data Augmentation ---")
print(f"Original  : {original}")
print(f"Augmented : {augmented}\n")
import numpy as np
from imblearn.over_sampling import SMOTE, RandomOverSampler
from collections import Counter

def balance_legal_data(X, y):
    counts = Counter(y)
    print(f"--- 2. Data Balancing ---")
    print(f"Original distribution : {counts}")
    
    # ตรวจสอบจำนวนตัวอย่างที่น้อยที่สุดในแต่ละคลาส
    min_samples = min(counts.values())
    
    if min_samples > 1:
        # ใช้ SMOTE (ปรับ k_neighbors ให้เหมาะสมกับจำนวนข้อมูลที่มี)
        k = min(5, min_samples - 1)
        sampler = SMOTE(k_neighbors=k, random_state=9)
    else:
        # หากมีข้อมูลเพียง 1 ตัวอย่าง ให้ใช้การสุ่มคัดลอก (Random Over Sampling)
        sampler = RandomOverSampler(random_state=9)
        
    X_res, y_res = sampler.fit_resample(X, y)
    print(f"Balanced distribution: {Counter(y_res)}\n")
    return X_res, y_res

# จำลองข้อมูล Imbalance (class 0 = 10 ตัวอย่าง, class 1 = 2 ตัวอย่าง)
X_mock = np.random.randn(12, 5)
y_mock = np.array([0]*10 + [1]*2)
X_res, y_res = balance_legal_data(X_mock, y_mock)
import torch
import torch.nn as nn

class LegalBiLSTM(nn.Module):
    def __init__(self, input_dim=16, hidden_dim=32, output_dim=3):
        super(LegalBiLSTM, self).__init__()
        # Bidirectional LSTM
        self.lstm = nn.LSTM(
            input_dim, 
            hidden_dim, 
            batch_first=True, 
            bidirectional=True
        )
        # FC Layer รับ input จาก 2 ทิศทาง (hidden_dim * 2)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)
        
        # Weight Initialization เพื่อความเสถียร
        nn.init.xavier_uniform_(self.fc.weight)

    def forward(self, x):
        # lstm_out shape: [batch, seq_len, hidden_dim * 2]
        lstm_out, _ = self.lstm(x)
        
        # Mean Pooling: หาค่าเฉลี่ยของทุก Token ในประโยค
        pooled = torch.mean(lstm_out, dim=1)
        
        return self.fc(pooled)

# ทดสอบการ Forward Pass
model = LegalBiLSTM()
# จำลอง Input: [1 ประโยค, 5 คำ, 16 มิติ (Embedding)]
sample_input = torch.randn(1, 5, 16) 
output = model(sample_input)

print(f"--- 3. BiLSTM Output ---")
print(f"Logits shape: {output.shape}")
print(f"Logits value: {output.detach().numpy()}")