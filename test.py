import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from tkinter import filedialog, Tk, Label, Button

# ---------- Settings ----------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMAGE_SIZE = 224
SUPPORT_PATH = "FEWSHOT_SUPPORT"
MODEL_PATH = "protonet_resnet_encoder.pth"

# ---------- Encoder ----------
class ResNetEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        from torchvision.models import resnet18
        backbone = resnet18(weights=None)
        self.encoder = nn.Sequential(*list(backbone.children())[:-1])
    
    def forward(self, x):
        x = self.encoder(x)
        return x.view(x.size(0), -1)

# ---------- Distance Function ----------
def euclidean_dist(a, b):
    n = a.size(0)
    m = b.size(0)
    a = a.unsqueeze(1).expand(n, m, -1)
    b = b.unsqueeze(0).expand(n, m, -1)
    return ((a - b) ** 2).sum(2)

# ---------- Support Set ----------
def load_support_set(path, transform):
    support_images = []
    support_labels = []
    label_map = {}
    idx = 0
    for cls in os.listdir(path):
        cls_path = os.path.join(path, cls)
        if not os.path.isdir(cls_path): continue
        label_map[idx] = cls
        for img_name in os.listdir(cls_path):
            if img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                img_path = os.path.join(cls_path, img_name)
                img = Image.open(img_path).convert("RGB")
                img = transform(img)
                support_images.append(img)
                support_labels.append(idx)
        idx += 1
    return torch.stack(support_images).to(DEVICE), torch.tensor(support_labels).to(DEVICE), label_map

# ---------- Prediction (Top-3) ----------
def predict_image(img_path):
    model.eval()
    img = Image.open(img_path).convert("RGB")
    img = transform(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        query_embed = model(img)                      # (1, D)
        support_embed = model(support_images)         # (N, D)
        prototypes = []

        for cls in torch.unique(support_labels):
            cls_embed = support_embed[support_labels == cls]
            prototype = cls_embed.mean(0)
            prototypes.append(prototype)

        prototypes = torch.stack(prototypes)          # (C, D)
        dists = euclidean_dist(query_embed, prototypes)  # (1, C)
        probs = F.softmax(-dists, dim=1)              # convert to probabilities

        top3_probs, top3_indices = torch.topk(probs, k=3, dim=1)
        top3_classes = [label_map[i.item()] for i in top3_indices[0]]
        top3_confidences = [top3_probs[0, i].item() * 100 for i in range(3)]

        # Combine class names with confidence
        result = "\n".join(
            f"{i+1}. {top3_classes[i]}"
            for i in range(3)
        )
        return result

# ---------- GUI ----------
def select_image():
    file_path = filedialog.askopenfilename(
        filetypes=[("Image Files", "*.jpg *.jpeg *.png")]
    )
    if file_path:
        prediction_result = predict_image(file_path)
        result_label.config(text=f"Top 3 Predicted Disease Classes:\n{prediction_result}")

# ---------- Transform ----------
transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
])

# ---------- Load Model and Support Set ----------
model = ResNetEncoder().to(DEVICE)
state_dict = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True)
model.load_state_dict(state_dict)
support_images, support_labels, label_map = load_support_set(SUPPORT_PATH, transform)

# ---------- Build GUI ----------
root = Tk()
root.title("Disease Classifier - Few Shot")
root.geometry("400x250")
root.configure(bg="#f0f0f0")

title_label = Label(root, text="Few-Shot Disease Classifier", font=("Arial", 14), bg="#f0f0f0")
title_label.pack(pady=10)

select_button = Button(root, text="Select Image", command=select_image, font=("Arial", 12), bg="#007acc", fg="white")
select_button.pack(pady=10)

result_label = Label(root, text="Prediction will appear here.", font=("Arial", 11), bg="#f0f0f0", justify="left")
result_label.pack(pady=10)

root.mainloop()


