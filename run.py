from mentalitystorm import OneShotEasyTrainer, config, BceKldLoss, MseKldLoss, OpenCV
import torchvision
import torchvision.transforms as TVT
import models

if __name__ == '__main__':

    dataset = torchvision.datasets.ImageFolder(
        root=config.datapath('spaceinvaders/images/dev'),
        transform=TVT.Compose([TVT.ToTensor()])
    )

    convolutions = models.ConvVAE4Fixed((210, 160))

    convolutions.registerView('z_corr', OpenCV('z_corr', (512, 512)))

    #loss = BceKldLoss()
    loss = MseKldLoss()
    loss = MseKldLoss(beta=3.0)
    easy = OneShotEasyTrainer()
    easy.run(convolutions, dataset, batch_size=20, epochs=20, lossfunc=loss)