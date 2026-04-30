import os
import random
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset


class FewShotEpisodicDataset:
    def __init__(self, root_dir, transform=None):
        self.transform = transform
        self.root_dir = root_dir

        # Build a dictionary: class_name -> list of image paths
        self.class_to_images = {
            cls: [
                os.path.join(root_dir, cls, img)
                for img in os.listdir(os.path.join(root_dir, cls))
                if img.lower().endswith(('.jpg', '.jpeg', '.png'))
            ]
            for cls in os.listdir(root_dir)
            if os.path.isdir(os.path.join(root_dir, cls))
        }

        # Filter classes with fewer than (k + q) images later
        self.all_classes = list(self.class_to_images.keys())

    def sample_episode(self, n_way=6, k_shot=5, q_query=1):
        # Filter out classes that don't have enough images
        valid_classes = [
            cls for cls in self.all_classes
            if len(self.class_to_images[cls]) >= (k_shot + q_query)
        ]
        assert len(valid_classes) >= n_way, f"Not enough valid classes for n={n_way}"

        # Sample n unique classes
        selected_classes = random.sample(valid_classes, n_way)

        support_set = []
        query_set = []
        labels = []

        for label_idx, cls in enumerate(selected_classes):
            images = random.sample(self.class_to_images[cls], k_shot + q_query)
            support_imgs = images[:k_shot]
            query_imgs = images[k_shot:]

            for img_path in support_imgs:
                img = Image.open(img_path).convert("RGB")
                if self.transform:
                    img = self.transform(img)
                support_set.append((img, label_idx))

            for img_path in query_imgs:
                img = Image.open(img_path).convert("RGB")
                if self.transform:
                    img = self.transform(img)
                query_set.append((img, label_idx))

            labels.append(cls)

        return support_set, query_set, labels


# ---------- 🔽 Example usage ----------

if __name__ == "__main__":
    import dibbu
    from torchvision.transforms import ToTensor, Resize, Compose

    # Update the path to your dataset
    dataset_path = r"C:\Users\RANBIR\Desktop\fewshot"

    # Basic transform (customize this as needed)
    transform = Compose([
        Resize((224, 224)),
        ToTensor()
    ])

    dataset = FewShotEpisodicDataset(dataset_path, transform=transform)

    # Define episodic parameters
    n = 6  # number of classes per episode
    k = 5  # number of support images per class
    q = 1  # number of query images per class

    # Sample one episode
    support, query, class_labels = dataset.sample_episode(n_way=n, k_shot=k, q_query=q)

    print(f"\n✅ Sampled Episode:\nClasses: {class_labels}")
    print(f"Support Set Size: {len(support)}")
    print(f"Query Set Size: {len(query)}")
    print(f"Support Sample Example: {support[0][0].shape}, Label: {support[0][1]}")
