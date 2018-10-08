from mentalitystorm.instrumentation import tb_test_loss_term, register_tb, LatentInstrument
from mentalitystorm.data import AutoEncodeSelect, StandardSelect, GymImageDataset
from mentalitystorm import config, ImageViewer, DataPackage, Run, SimpleRunFac, Params, Handles, transforms
from mentalitystorm.losses import MSELoss
import torchvision
import torchvision.transforms as TVT
from models import Compressor
from tqdm import tqdm
from torch.optim import Adam

if __name__ == '__main__':

    input_viewer = ImageViewer('input', (320, 480))
    output_viewer = ImageViewer('output', (320, 480))
    latent_viewer = ImageViewer('latent', (320, 480))
    latent_instr = LatentInstrument()

    invaders = torchvision.datasets.ImageFolder(
        root=config.datapath('spaceinvaders/images/raw'),
        transform=TVT.Compose([TVT.ToTensor()])
    )

    from mentalitystorm.transforms import ColorMask
    shots = ColorMask(lower=[128, 128, 128], upper=[255, 255, 255])

    cartpole = torchvision.datasets.ImageFolder(
        root=config.datapath('cartpole/images/raw'),
        transform=TVT.Compose([TVT.ToTensor()])
    )

    co_ord_conv_invaders = GymImageDataset(directory=config.datapath(r'SpaceInvaders-v4\images\raw_v1\all'),
                                           input_transform=TVT.Compose([TVT.ToTensor(), transforms.CoordConv()]),
                                           target_transform=TVT.Compose([TVT.ToTensor()]))

    co_ord_conv_invaders_w_target = GymImageDataset(directory=config.datapath(r'SpaceInvaders-v4\images\raw_v1\all'),
                                                    input_transform=TVT.Compose(
                                                        [TVT.ToTensor(), transforms.CoordConv()]),
                                                    target_transform=TVT.Compose(
                                                        [TVT.ToTensor(), transforms.CoordConv()]))

    regular_invaders = GymImageDataset(directory=config.datapath(r'SpaceInvaders-v4\images\raw_v1\all'),
                                       input_transform=TVT.Compose([TVT.ToTensor()]),
                                       target_transform=TVT.Compose([TVT.ToTensor()]))

    co_ord_conv_data_package = DataPackage(co_ord_conv_invaders, StandardSelect())
    control_data_package = DataPackage(regular_invaders, AutoEncodeSelect())

    run_fac = SimpleRunFac()
    #model = Params(AtariVAE2DLatent, (210, 160), 32, input_channels=5, output_channels=3)
    compressor = Params(Compressor, (210, 160), 32, input_channels=5, output_channels=3)
    #model = Params(ConvVAE4Fixed, (400, 600), 2)

    opt = Params(Adam, lr=1e-3)
    #run_fac.run_list.append(Run(model, opt, Params(BceKldLoss), control_data_package, run_name='control BCE'))
    run_fac.run_list.append(Run(compressor, opt, Params(MSELoss), co_ord_conv_data_package, run_name='biglatent MSE with co-conv'))

    #run_fac = SimpleRunFac.resume(r'C:\data\runs\489', co_ord_conv_data_package)

    batch_size = 64
    epochs = 100

    for model, opt, loss_fn, data_package, trainer, tester, run in run_fac:
        dev, train, test, selector = data_package.loaders(batch_size=batch_size)

        model.register_forward_hook(input_viewer.view_input)
        model.decoder.register_forward_hook(latent_viewer.view_input)
        model.register_forward_hook(output_viewer.view_output)
        register_tb(run)

        for epoch in tqdm(run.for_epochs(epochs), 'epochs', epochs):

            #epoch.register_after_hook(write_histogram)
            epoch.execute_before(epoch)

            trainer.train(model, opt, loss_fn, train, selector, run, epoch)

            handles = Handles()
            handles += loss_fn.register_hook(tb_test_loss_term)
            tester.test(model, loss_fn, test, selector, run, epoch)
            handles.remove()
            epoch.execute_after(epoch)
            run.save()
