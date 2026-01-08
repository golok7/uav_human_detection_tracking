from detection_dataset import HumanDetectionDataset

dataset = HumanDetectionDataset(
    images_dir="dataset/train/images",
    labels_dir="dataset/train/labels"
)

image, boxes = dataset[0]

print("Image shape:", image.shape)
print("Boxes:", boxes)
print("Number of humans:", len(boxes))
