import torch
from model import CircleNet
from trainer import Trainer
from dataloader import SphereDataset
from torch.utils.data import DataLoader


def main():
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	model = CircleNet().to(device)

	from torchvision import transforms

	transform = transforms.Compose([
	transforms.ToTensor(),
	])

	train_dataset = SphereDataset(csv_file='labels.csv', root_dir='data/train/', transform=transform)
	train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=0)
        
	val_dataset = SphereDataset(csv_file='labels.csv', root_dir='data/val/', transform=transform)
	val_loader = DataLoader(val_dataset, batch_size=8, shuffle=True, num_workers=0)

	trainer = Trainer(model, device, train_loader, val_loader)
	trainer.train(num_epochs=50)

if __name__ == '__main__':
    main()