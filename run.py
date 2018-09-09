from mentalitystorm import OneShotEasyTrainer, config, BceKldLoss, MseKldLoss
from pathlib import Path
import torchvision
import torchvision.transforms as TVT
import models

if __name__ == '__main__':

    dataset = torchvision.datasets.ImageFolder(
        root=config.datapath('spaceinvaders/images/raw'),
        transform=TVT.Compose([TVT.ToTensor()])
    )

    convolutions = models.ConvVAE4Fixed((210, 160))

    #loss = BceKldLoss()
    loss = MseKldLoss()
    easy = OneShotEasyTrainer()
    easy.run(convolutions, dataset, batch_size=50, epochs=20, lossfunc=loss)