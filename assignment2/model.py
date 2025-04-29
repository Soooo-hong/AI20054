import pytorch_lightning as pl
import torch.nn.functional as F
import torch.optim as optim
import torchvision.models as models
import torch.nn as nn

class LitResNet(pl.LightningModule):
    def __init__(self, learning_rate=0.001):
        super().__init__()
        self.model = models.resnet50(pretrained=False)
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, 10)  # CIFAR-10은 10클래스
        self.learning_rate = learning_rate

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        self.log('val_loss', loss)
        return loss
    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        self.log('test_loss', loss)
        acc = (y_hat.argmax(dim=1) == y).float().mean()
        self.log('test_acc', acc)
        return loss
    

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer