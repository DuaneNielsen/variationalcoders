from mentalitystorm.instrumentation import tb_test_loss_term, register_tb, LatentInstrument
from mentalitystorm.data import StandardSelect, GymImageDataset
from mentalitystorm.config import config
from mentalitystorm.observe import ImageViewer
from mentalitystorm.data import DataPackage
from mentalitystorm.runners import SimpleRunFac, Params
from mentalitystorm.util import Handles
import mentalitystorm.transforms as tf

import torchvision.transforms as TVT
from models import Compressor
from tqdm import tqdm
from torch.optim import Adam

if __name__ == '__main__':

    input_viewer = ImageViewer('input', (320, 480))
    output_viewer = ImageViewer('output', (320, 480))
    latent_viewer = ImageViewer('latent', (320, 480))
    latent_instr = LatentInstrument()

    shots = tf.ColorMask(lower=[128, 128, 128], upper=[255, 255, 255], append=False)

    co_ord_conv_shots = GymImageDataset(directory=config.datapath(r'SpaceInvaders-v4\images\raw_v1\all'),
                                        input_transform=TVT.Compose([shots, TVT.ToTensor(), tf.CoordConv()]),
                                        target_transform=TVT.Compose([shots, TVT.ToTensor()]))

    co_ord_conv_data_package = DataPackage(co_ord_conv_shots, StandardSelect())

    #run_fac = SimpleRunFac()
    compressor = Params(Compressor, (210, 160), 1, input_channels=3, output_channels=1)

    opt = Params(Adam, lr=1e-3)
    #run_fac.run_list.append(Run(compressor, opt, Params(MSELoss), co_ord_conv_data_package, run_name='shots_v1'))

    run_fac = SimpleRunFac.resume(r'C:\data\runs\549', co_ord_conv_data_package)
    batch_size = 64
    epochs = 30

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
