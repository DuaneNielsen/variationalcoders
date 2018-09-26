from mentalitystorm import Storeable, config, Demo, MseKldLoss, OpenCV
import torchvision
import torchvision.transforms as TVT

if __name__ == '__main__':

    dataset = torchvision.datasets.ImageFolder(
        root=config.datapath('spaceinvaders/images/raw'),
        transform=TVT.Compose([TVT.ToTensor()])
    )

    convolutions = Storeable.load(r'C:\data\runs\399\B-VAE loss 2.0\epoch0004')

    # todo demo of effect of each z parameter
    demo = Demo()
    convolutions.registerView('z_corr', OpenCV('z_corr', (512, 512)))
    #lossfunc = MseKldLoss()
    #demo.test(convolutions, dataset, 128, lossfunc)
    demo.rotate(convolutions, 2)
    #demo.sample(convolutions, 2, samples=20)
    #demo.demo(convolutions, dataset)