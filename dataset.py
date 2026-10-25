import os
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch
from sklearn.preprocessing import StandardScaler

class FoodDataset(Dataset):
    def __init__(self, df, encoded_ingrs, transform=None):
        self.df = df.reset_index(drop=True)
        self.encoded_ingrs = encoded_ingrs
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        dish_id = row['dish_id']

        img_path = os.path.join("data", "images", str(dish_id), "rgb.png")
        
        if os.path.exists(img_path):
            image = Image.open(img_path).convert("RGB")
        else:
            image = Image.new("RGB", (224, 224), (0, 0, 0))
            
        image_np = np.array(image)

        if self.transform:
            augmented = self.transform(image=image_np)
            image_tensor = augmented['image']

        # Масса и multi-hot вектор состава из готовых данных
        mass = torch.tensor([row['scaled_mass']], dtype=torch.float32)
        ingr_features = torch.tensor(self.encoded_ingrs[idx], dtype=torch.float32)
        target = torch.tensor(row['total_calories'], dtype=torch.float32)
        
        return image_tensor, ingr_features, mass, target

def get_dataloaders(batch_size):
    # Читаем файл с текстовыми ингредиентами
    dish_df = pd.read_csv(os.path.join("data", "dish_processed.csv"))
    ingr_df = pd.read_csv(os.path.join("data", "ingredients.csv"))

    # Составляем список уникальных текстовых названий продуктов
    unique_ingredients = sorted(ingr_df['ingr'].unique())
    ingr_to_idx = {name: idx for idx, name in enumerate(unique_ingredients)}
    num_ingredients = len(unique_ingredients)

    # Собираем Multi-Hot векторы
    encoded_ingrs_list = []
    for _, row in dish_df.iterrows():
        vector = np.zeros(num_ingredients, dtype=np.float32)
        if not pd.isna(row['ingredients_clean']):
            # Просто бьем готовую строку "salmon, zucchini" по запятой
            items = [item.strip() for item in str(row['ingredients_clean']).split(',')]
            for item in items:
                if item in ingr_to_idx:
                    vector[ingr_to_idx[item]] = 1.0
        encoded_ingrs_list.append(vector)
    encoded_ingrs = np.vstack(encoded_ingrs_list)

    # Масштабируем массу порции
    scaler = StandardScaler()
    dish_df['scaled_mass'] = scaler.fit_transform(dish_df[['total_mass']]).astype(np.float32)
    
    # Разделяем по метке split
    train_mask = dish_df['split'] == 'train'
    test_mask = dish_df['split'] == 'test'
    
    train_df = dish_df[train_mask].reset_index(drop=True)
    test_df = dish_df[test_mask].reset_index(drop=True)

    train_encoded = encoded_ingrs[train_mask.values]
    test_encoded = encoded_ingrs[test_mask.values]
    
    # Трансформации albumentations
    train_transforms = A.Compose([
        A.Resize(224, 224),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])

    test_transforms = A.Compose([
        A.Resize(224, 224),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])
    
    train_dataset = FoodDataset(train_df, train_encoded, transform=train_transforms)
    test_dataset = FoodDataset(test_df, test_encoded, transform=test_transforms)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    
    return train_loader, test_loader, num_ingredients
