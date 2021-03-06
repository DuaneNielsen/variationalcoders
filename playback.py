from mentalitystorm.instrumentation import register_tb, LatentInstrument
from mentalitystorm.data import StandardSelect, GymImageDataset
from mentalitystorm import config, ImageViewer, DataPackage, Run, SimpleRunFac, Params, transforms as tf
import torchvision.transforms as TVT
from tqdm import tqdm
from torch.optim import Adam
from mentalitystorm.basemodels import MultiChannelAE, DummyAE
from mentalitystorm.train import SimpleInference

if __name__ == '__main__':

    input_viewer = ImageViewer('input', (320, 480))
    output_viewer = ImageViewer('output', (320, 480))
    latent_viewer = ImageViewer('latent', (320, 480))
    latent_instr = LatentInstrument()

    shots = tf.ColorMask(lower=[128, 128, 128], upper=[255, 255, 255], append=True)
    player = tf.ColorMask(lower=[30, 100, 40], upper=[70, 180, 70], append=True)
    cut = tf.SetRange(0, 60, 0, 210, [4])
    select = tf.SelectChannels([3, 4])

    segmentor = TVT.Compose([shots, player, cut, select, TVT.ToTensor(), tf.CoordConv()])

    co_ord_conv_shots = GymImageDataset(directory=config.datapath(r'SpaceInvaders-v4\images\raw_v1\all'),
                                        input_transform=segmentor,
                                        target_transform=segmentor)

    co_ord_conv_data_package = DataPackage(co_ord_conv_shots, StandardSelect())

    channel_coder = Params(MultiChannelAE)
    opt = Params(Adam, lr=1e-3)

    run_fac = SimpleRunFac()
    run_fac.run_list.append(Run(channel_coder, None, None, co_ord_conv_data_package,
                                run_name='shots_v1', trainer=SimpleInference()))

    #run_fac = SimpleRunFac.resume(r'C:\data\runs\549', co_ord_conv_data_package)
    batch_size = 64
    epochs = 30

    for model, opt, loss_fn, data_package, trainer, tester, run in run_fac:
        dev, train, test, selector = data_package.loaders(batch_size=batch_size)

        model.add_ae(DummyAE(), [0, 1, 2, 3])
        model.add_ae(DummyAE(), [0, 2, 3])

        model.register_forward_hook(input_viewer.view_input)
        model.register_forward_hook(output_viewer.view_output)
        register_tb(run)

        for epoch in tqdm(run.for_epochs(epochs), 'epochs', epochs):

            epoch.execute_before(epoch)
            trainer.infer(model, loss_fn, test, selector, run, epoch)

            epoch.execute_after(epoch)
            run.save()