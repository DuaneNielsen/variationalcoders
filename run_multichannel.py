from mentalitystorm.instrumentation import register_tb, LatentInstrument
from mentalitystorm.data import StandardSelect, GymImageDataset, DataPackage
from mentalitystorm.config import config
from mentalitystorm.observe import ImageViewer
from mentalitystorm.runners import Run, SimpleRunFac, Params
import mentalitystorm.transforms as tf
import torchvision.transforms as TVT
from tqdm import tqdm
from torch.optim import Adam
from mentalitystorm.basemodels import MultiChannelAE
from mentalitystorm.train import SimpleInference

if __name__ == '__main__':

    input_viewer = ImageViewer('input', (320, 480), channels=[0, 1, 2])
    input2_viewer = ImageViewer('input2', (320, 480), channels=[3, 4, 5])
    shot_viewer = ImageViewer('shot', (320, 480), channels=[0])
    player_viewer = ImageViewer('player', (320, 480), channels=[1])
    invader_viewer = ImageViewer('invader', (320, 480), channels=[2])
    barrier_viewer = ImageViewer('barrier', (320, 480), channels=[3])
    decode_viewer = ImageViewer('decoded', (320, 480), channels=[0, 1, 2])
    decode2_viewer = ImageViewer('decoded2', (320, 480), channels=[3])
    latent_instr = LatentInstrument()

    def view_decode(model, input, output):
        image = model.decode(output)
        decode_viewer.update(image)
        decode2_viewer.update(image)


    shots = tf.ColorMask(lower=[128, 128, 128], upper=[255, 255, 255], append=True)
    player = tf.ColorMask(lower=[30, 100, 40], upper=[70, 180, 70], append=True)
    cut_player = tf.SetRange(0, 60, 0, 210, [4])
    invader = tf.ColorMask(lower=[120, 125, 30], upper=[140, 140, 130], append=True)
    cut_invader = tf.SetRange(0, 30, 0, 210, [5])
    barrier = tf.ColorMask(lower=[120, 74, 30], upper=[190, 100, 70], append=True)
    select = tf.SelectChannels([3, 4, 5, 6])

    segmentor = TVT.Compose([shots, player, cut_player, invader, cut_invader,
                             barrier, select, TVT.ToTensor(), tf.CoordConv()])

    co_ord_conv_shots = GymImageDataset(directory=config.datapath(r'SpaceInvaders-v4\images\raw_v1\all'),
                                        input_transform=segmentor,
                                        target_transform=segmentor)

    co_ord_conv_data_package = DataPackage(co_ord_conv_shots, StandardSelect())

    channel_coder = Params(MultiChannelAE)
    opt = Params(Adam, lr=1e-3)

    run_fac = SimpleRunFac(increment_run=False)
    run_fac.run_list.append(Run(channel_coder, None, None, co_ord_conv_data_package,
                                run_name='shots_v1', trainer=SimpleInference()))

    #run_fac = SimpleRunFac.resume(r'C:\data\runs\549', co_ord_conv_data_package)
    batch_size = 1
    epochs = 30

    shot_encoder = Run.load_model(r'c:\data\runs\549\shots_v1\epoch0060.run').eval().to(device=config.device())
    player_encoder = Run.load_model(r'c:\data\runs\580\shots_v1\epoch0081.run').eval().to(device=config.device())
    invader_encoder = Run.load_model(r'c:\data\runs\587\shots_v1\epoch0030.run').eval().to(device=config.device())
    barrier_encoder = Run.load_model(r'c:\data\runs\588\barrier\epoch0019.run').eval().to(device=config.device())

    for model, opt, loss_fn, data_package, trainer, tester, run in run_fac:
        dev, train, test, selector = data_package.loaders(batch_size=batch_size)

        model.add_ae(shot_encoder, [0, 4, 5], [0])
        model.add_ae(player_encoder, [1, 4, 5], [1])
        model.add_ae(invader_encoder, [2, 4, 5], [2])
        model.add_ae(barrier_encoder, [3, 4, 5], [3])

        model.register_forward_hook(input_viewer.view_input)
        model.register_forward_hook(input2_viewer.view_input)
        model.register_forward_hook(shot_viewer.view_output)
        model.register_forward_hook(player_viewer.view_output)
        model.register_forward_hook(invader_viewer.view_output)
        model.register_forward_hook(barrier_viewer.view_output)
        model.register_forward_hook(view_decode)
        register_tb(run)

        for epoch in tqdm(run.for_epochs(epochs), 'epochs', epochs):

            epoch.execute_before(epoch)
            trainer.infer(model, loss_fn, test, selector, run, epoch)

            epoch.execute_after(epoch)
            run.save()
