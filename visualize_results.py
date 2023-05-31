import torch
from model import CircleNet
from dataloader import SphereDataset
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

def visualize_prediction(model, dataloader):
    fig, axs = plt.subplots(3, 3, figsize=(15, 15))

    for i, (image, label) in enumerate(dataloader):
        image = image.to(device, dtype=torch.float32)
        model = model.eval()
        with torch.no_grad():
            output = model(image).squeeze() 

        image = image.squeeze().cpu().numpy()
        output = output.cpu().numpy()
        label = label.numpy()

        row = i // 3
        col = i % 3

        axs[row, col].imshow(image, cmap='gray')
        axs[row, col].plot(label[0][0], label[0][1], 'go', label='True center')
        axs[row, col].plot(output[0], output[1], 'ro', label='Predicted center')
        axs[row, col].legend()

        if i == 8:
            break

    plt.savefig('predictions.png')

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CircleNet()
    model.to(device)
    model.load_state_dict(torch.load('best_model.pth'))

    from torchvision import transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    dataset = SphereDataset(csv_file='labels.csv', root_dir='data/val/', transform=transform)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    visualize_prediction(model, dataloader)
