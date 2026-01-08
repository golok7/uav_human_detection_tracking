from dataloader import get_dataloaders

def main():
    train_loader, val_loader, test_loader, classes = get_dataloaders()

    print("Classes:", classes)
    print("Train batches:", len(train_loader))
    print("Validation batches:", len(val_loader))
    print("Test batches:", len(test_loader))

    images, labels = next(iter(train_loader))
    print("Image batch shape:", images.shape)
    print("Label batch shape:", labels.shape)

if __name__ == "__main__":
    main()
