from mentalitystorm import Storeable, config, Demo
import torchvision
import torchvision.transforms as TVT

if __name__ == '__main__':

    dataset = torchvision.datasets.ImageFolder(
        root=config.datapath('spaceinvaders/images/raw'),
        transform=TVT.Compose([TVT.ToTensor()])
    )

    convolutions = Storeable.load('K3Q3OLTBM2K0NI0S')

    demo = Demo()
    demo.demo(convolutions, dataset)