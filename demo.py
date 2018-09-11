from mentalitystorm import Storeable, config, Demo, MseKldLoss, OpenCV
import torchvision
import torchvision.transforms as TVT

if __name__ == '__main__':

    dataset = torchvision.datasets.ImageFolder(
        root=config.datapath('spaceinvaders/images/raw'),
        transform=TVT.Compose([TVT.ToTensor()])
    )

    convolutions = Storeable.load('GM53H301W5YS38XH')

    # todo demo of effect of each z parameter
    demo = Demo()
    convolutions.registerView('z_corr', OpenCV('z_corr', (512, 512)))
    #lossfunc = MseKldLoss()
    #demo.test(convolutions, dataset, 128, lossfunc)
    demo.rotate(convolutions, 16)
    demo.sample(convolutions, 16, samples=20)
    demo.demo(convolutions, dataset)