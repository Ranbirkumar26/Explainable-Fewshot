import os
import random
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.models import resnet18
from torch.utils.data import Dataset, DataLoader

# ---------- Config ----------
N_WAY = 6
K_SHOT = 5
Q_QUERY = 1
EPISODES_PER_EPOCH = 100
EPISODES_PER_BATCH = 4
DATASET_PATH = r"C:\Users\RANBIR\Desktop\fewshot"
DEVICE = torch.device("cuda" if torch.cuda.is_available())

# ---------- Transform ----------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor()
])

# ---------- Episodic Sampler ----------
class FewShotDataset:
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.class_to_images = {
            cls: [
                os.path.join(root_dir, cls, img)
                for img in os.listdir(os.path.join(root_dir, cls))
                if img.lower().endswith(('.jpg', '.png', '.jpeg'))
            ]
            for cls in os.listdir(root_dir)
            if os.path.isdir(os.path.join(root_dir, cls))
        }
        self.all_classes = list(self.class_to_images.keys())

    def sample_episode(self, n_way, k_shot, q_query):
        valid_classes = [cls for cls in self.all_classes if len(self.class_to_images[cls]) >= (k_shot + q_query)]
        selected_classes = random.sample(valid_classes, n_way)

        support_set, query_set = [], []

        for label, cls in enumerate(selected_classes):
            images = random.sample(self.class_to_images[cls], k_shot + q_query)
            support_imgs = images[:k_shot]
            query_imgs = images[k_shot:]

            for img_path in support_imgs:
                img = Image.open(img_path).convert('RGB')
                if self.transform:
                    img = self.transform(img)
                support_set.append((img, label))

            for img_path in query_imgs:
                img = Image.open(img_path).convert('RGB')
                if self.transform:
                    img = self.transform(img)
                query_set.append((img, label))

        return support_set, query_set

# ---------- Encoder: ResNet18 ----------
class ResNetEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        backbone = resnet18(pretrained=True)
        self.encoder = nn.Sequential(*list(backbone.children())[:-1])  # remove FC
        self.out_dim = 512

    def forward(self, x):
        x = self.encoder(x)
        return x.view(x.size(0), -1)

# ---------- Distance ----------
def euclidean_dist(x, y):
    n = x.size(0)
    m = y.size(0)
    x = x.unsqueeze(1).expand(n, m, -1)
    y = y.unsqueeze(0).expand(n, m, -1)
    return torch.pow(x - y, 2).sum(2)

# ---------- Training Loop ----------
def train(dataset, model, optimizer):
    criterion = nn.CrossEntropyLoss()
    model.train()

    for episode in range(EPISODES_PER_EPOCH):
        total_loss = 0.0
        total_acc = 0.0

        for _ in range(EPISODES_PER_BATCH):
            support_set, query_set = dataset.sample_episode(N_WAY, K_SHOT, Q_QUERY)

            support_imgs = torch.stack([img for img, _ in support_set]).to(DEVICE)
            support_labels = torch.tensor([label for _, label in support_set]).to(DEVICE)
            query_imgs = torch.stack([img for img, _ in query_set]).to(DEVICE)
            query_labels = torch.tensor([label for _, label in query_set]).to(DEVICE)

            # Encode
            support_embeds = model(support_imgs)
            query_embeds = model(query_imgs)

            # Compute class prototypes
            support_embeds = support_embeds.view(N_WAY, K_SHOT, -1).mean(1)
            query_embeds = query_embeds.view(len(query_set), -1)

            dists = euclidean_dist(query_embeds, support_embeds)
            preds = -dists
            loss = criterion(preds, query_labels)
            acc = (preds.argmax(1) == query_labels).float().mean().item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_acc += acc

        avg_loss = total_loss / EPISODES_PER_BATCH
        avg_acc = total_acc / EPISODES_PER_BATCH

        print(f"[Episode {episode + 1}/{EPISODES_PER_EPOCH}] Loss: {avg_loss:.4f}, Accuracy: {avg_acc * 100:.2f}%")


# ---------- Main ----------
if __name__ == "__main__":
    dataset = FewShotDataset(DATASET_PATH, transform=transform)
    model = ResNetEncoder().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    train(dataset, model, optimizer)

    # Save encoder after training
    torch.save(model.state_dict(), "protonet_resnet_encoder.pth")
    print("✅ Model saved as protonet_resnet_encoder.pth")

    #test_dataset = FewShotDataset("C:/Users/RANBIR/Desktop/fewshot/FEWSHOT_TEST", transform=transform)
    #test(test_dataset, model, episodes=20)


