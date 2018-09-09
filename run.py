from mentalitystorm import OneShotEasyRunner, Config, BceKldLoss
from pathlib import Path
import torchvision
import torchvision.transforms as TVT
import models

if __name__ == '__main__':

    datadir = Path(Config().DATA_PATH) / 'spaceinvaders/images/raw'
    dataset = torchvision.datasets.ImageFolder(
        root=datadir.absolute(),
        transform=TVT.Compose([TVT.ToTensor()])
    )

    convolutions = models.ConvVAE4Fixed((210, 160))

    loss = BceKldLoss()
    easy = OneShotEasyRunner()
    easy.run(convolutions, dataset, batch_size=128, epochs=20, lossfunc=loss)