import os
import random
import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
import torchmetrics
import timm
from torch.optim.lr_scheduler import ReduceLROnPlateau

from dataset import get_dataloaders

class Config:
    def __init__(self):
        self.SEED = 42
        self.BATCH_SIZE = 4
        self.EPOCHS = 10
        self.IMAGE_LR = 1e-5       
        self.TEXT_LR = 5e-4        
        self.CLASSIFIER_LR = 1e-3  
        self.WEIGHT_DECAY = 1e-4
        self.PROJECTION_DIM = 256
        self.MODEL_SAVE_PATH = os.path.join("data", "best_calorie_model.pth")

def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class CalorieFusionModel(nn.Module):
    def __init__(self, num_ingredients, projection_dim=256):
        super(CalorieFusionModel, self).__init__()

        self.image_encoder = timm.create_model('efficientnet_b0', pretrained=True, num_classes=0)
        in_features = self.image_encoder.num_features
        
        self.image_projection = nn.Sequential(
            nn.Linear(in_features, projection_dim),
            nn.BatchNorm1d(projection_dim),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        self.text_encoder = nn.Sequential(
            nn.Linear(num_ingredients, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

        self.text_projection = nn.Sequential(
            nn.Linear(256, projection_dim),
            nn.BatchNorm1d(projection_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        combined_dim = projection_dim + projection_dim + 1
        
        # Один выходной нейрон без функции активации
        self.classifier = nn.Sequential(
            nn.Linear(combined_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1)
        )

        def forward(self, image, text, mass):
        img_feats = self.image_encoder(image)
        img_proj = self.image_projection(img_feats)
        
        text_feats = self.text_encoder(text)
        text_proj = self.text_projection(text_feats)
        
        combined = torch.cat([img_proj, text_proj, mass], dim=1)
        output = self.classifier(combined)
        return output.squeeze(-1)

    def train(cfg):
    seed_everything(cfg.SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, test_loader, num_ingredients = get_dataloaders(cfg.BATCH_SIZE)
    model = CalorieFusionModel(num_ingredients=num_ingredients, projection_dim=cfg.PROJECTION_DIM).to(device)
    
    # Оптимизируем MAE (L1Loss) напрямую
    criterion = nn.L1Loss()
    
    train_mae_metric = torchmetrics.MeanAbsoluteError().to(device)
    test_mae_metric = torchmetrics.MeanAbsoluteError().to(device)
    
    # Единый список весов с раздельными LR
    optimizer = AdamW([
        {"params": model.text_encoder.parameters(), "lr": cfg.TEXT_LR},
        {"params": model.image_encoder.parameters(), "lr": cfg.IMAGE_LR},
        {"params": model.text_projection.parameters(), "lr": cfg.CLASSIFIER_LR},
        {"params": model.image_projection.parameters(), "lr": cfg.CLASSIFIER_LR},
        {"params": model.classifier.parameters(), "lr": cfg.CLASSIFIER_LR},
    ], weight_decay=cfg.WEIGHT_DECAY)
    
    scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=2, factor=0.5)
    best_mae = float('inf')

    for epoch in range(cfg.EPOCHS):
        model.train()
        train_mae_metric.reset()
        
        for images, texts, masses, targets in train_loader:
            images = images.to(device)
            texts = texts.to(device)
            masses = masses.to(device)
            targets = targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(images, texts, masses)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            train_mae_metric.update(outputs, targets)
            
        epoch_train_mae = train_mae_metric.compute().item()
        
        # Тестирование
        model.eval()
        test_mae_metric.reset()
        
        with torch.no_grad():
            for images, texts, masses, targets in test_loader:
                images = images.to(device)
                texts = texts.to(device)
                masses = masses.to(device)
                targets = targets.to(device)

                outputs = model(images, texts, masses)
                test_mae_metric.update(outputs, targets)
                
        epoch_test_mae = test_mae_metric.compute().item()
        scheduler.step(epoch_test_mae)
        
        print(f"Эпоха [{epoch+1}/{cfg.EPOCHS}] | Train MAE: {epoch_train_mae:.2f} | Test MAE: {epoch_test_mae:.2f}")
        
        if epoch_test_mae < best_mae:
            best_mae = epoch_test_mae
            torch.save(model.state_dict(), cfg.MODEL_SAVE_PATH)
            print(f"Успешно сохранено: {cfg.MODEL_SAVE_PATH}")
            
    print(f"Лучшая MAE: {best_mae:.2f}")
    return best_mae
