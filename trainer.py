import torch

class Trainer:
    def __init__(self, model, device, train_dataloader, valid_dataloader=None):
        self.model = model
        self.device = device
        self.train_dataloader = train_dataloader
        self.valid_dataloader = valid_dataloader
        self.criterion = torch.nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        
    def train_one_epoch(self):
        self.model.train()
        running_loss = 0.0
        for i, (images, labels) in enumerate(self.train_dataloader):
            images = images.to(self.device)
            labels = labels.to(self.device)

            # Forward pass
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)

            # Backward and optimize
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()
            
        return running_loss / len(self.train_dataloader)
    
    def validate_one_epoch(self):
        if not self.valid_dataloader:
            return None

        self.model.eval()
        running_loss = 0.0
        with torch.no_grad():
            for images, labels in self.valid_dataloader:
                images = images.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(images)
                #print(outputs[0], labels[0])
                loss = self.criterion(outputs, labels)

                running_loss += loss.item()

        return running_loss / len(self.valid_dataloader)

    def train(self, num_epochs, save_path='best_model.pth'):
        best_loss = float('inf')
        for epoch in range(num_epochs):
            train_loss = self.train_one_epoch()
            valid_loss = self.validate_one_epoch()

            if self.valid_dataloader:
                print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Valid Loss: {valid_loss:.4f}')
            else:
                print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}')

            # save the best model
            if valid_loss and valid_loss < best_loss:
                best_loss = valid_loss
                torch.save(self.model.state_dict(), save_path)

