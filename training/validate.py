import torch
import torch.nn as nn

def validate(model, loader, device):
    model.eval()

    criterion = nn.BCEWithLogitsLoss()

    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels, _ in loader:
            images = images.to(device)
            labels = labels.to(device)

            logits, _ = model(images)

            loss = criterion(
                logits.squeeze(1),
                labels.float()
            )

            running_loss += loss.item()

            preds = (torch.sigmoid(logits) > 0.5).long().squeeze(1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    avg_loss = running_loss / len(loader)
    accuracy = correct / total

    return avg_loss, accuracy
