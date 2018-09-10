from mentalitystorm import Storeable, config, Demo
import torchvision
import torchvision.transforms as TVT

if __name__ == '__main__':

    dataset = torchvision.datasets.ImageFolder(
        root=config.datapath('spaceinvaders/images/raw'),
        transform=TVT.Compose([TVT.ToTensor()])
    )

    convolutions = Storeable.load('WHR8K60HZMVFENSG')

    demo = Demo()
    demo.sample(convolutions, 32, samples=20)
    demo.demo(convolutions, dataset)