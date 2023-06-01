import torch
from model import CircleNet
from trainer import Trainer
from dataloader import SphereDataset
from torch.utils.data import DataLoader
from torchvision import transforms
import argparse


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('--n_epochs', type=int, default=50, help='Number of epochs for training')
	parser.add_argument('--batch_size', type=int, default=8, help='Batch size for training')
	parser.add_argument('--save_as_onnx', type=bool, default=False, help='Save model as onnx')
	args = parser.parse_args()
	
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	model = CircleNet().to(device)
	n_epochs = args.n_epochs
	batch_size = args.batch_size
	save_as_onnx = args.save_as_onnx

	transform = transforms.Compose([
	transforms.ToTensor(),
	])

	train_dataset = SphereDataset(csv_file='labels.csv', root_dir='data/train/', transform=transform)
	train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
        
	val_dataset = SphereDataset(csv_file='labels.csv', root_dir='data/val/', transform=transform)
	val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

	trainer = Trainer(model, device, train_loader, val_loader)
	trainer.train(num_epochs=n_epochs)
      
	if save_as_onnx:
		torch.onnx.export(model, torch.randn(1, 1, 96, 96).to(device), "model.onnx", verbose=False)

if __name__ == '__main__':
    main()