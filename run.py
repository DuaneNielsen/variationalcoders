from mentalitystorm import config, MseKldLoss, OpenCV, DataPackage, Selector, Run, RunFac, SimpleTrainer, SimpleTester,\
    SimpleRunFac, Params, Handles, LoadModel
import torchvision
import torch
import torchvision.transforms as TVT
from models import ConvVAE4Fixed
from tqdm import tqdm
import numpy as np
from torch.optim import Adam


class AutoEncodeSelect(Selector):
    def get_input(self, package, device):
        return package[0].to(device),

    def get_target(self, package, device):
        return package[0].to(device),


def print_loss_term(key, value):
    print('%s %f' % (key, value.item()))


def tb_train_loss_term(loss, loss_term, value):
    loss.run.tb.add_scalar('loss/train/' + loss_term, value.item(), loss.run.step)


def tb_test_loss_term(loss, loss_term, value):
    loss.run.tb.add_scalar('loss/test/' + loss_term, value.item(), loss.run.step)


def print_loss(args):
    print('Loss %f' % args.loss.item())


def tb_train_loss(args):
    args.run.tb.add_scalar('loss/Train Loss', args.loss.item(), args.run.step)


def tb_test_loss(args):
    args.run.tb.add_scalar('loss/Test Loss', args.loss.item(), args.run.step)


def tb_image(args):
    if args.run.step % 200 == 0:
        input_image = args.input_data[0][0].data
        output_image = args.output_data[0][0].data
        args.run.tb.add_image('input', input_image, args.run.step)
        args.run.tb.add_image('output', output_image, args.run.step)


def write_correlation(epoch):
    z = np.concatenate(epoch.context['zl'], axis=0).squeeze()
    corr = np.corrcoef(z, rowvar=False)
    corr = np.absolute(corr)
    epoch.run.tb.add_scalar('z/z_ave_correlation', (corr - np.identity(corr.shape[0])).mean(), epoch.run.step)
    corr = np.expand_dims(corr, axis=0)
    corr_viewer.update(corr)
    epoch.run.tb.add_image('corr_matrix', corr, run.step)


def write_histogram(epoch):
    z = np.concatenate(epoch.context['zl'], axis=0).squeeze()
    histograms = np.rollaxis(z, 1)
    for i, histogram in enumerate(histograms):
        epoch.run.tb.add_histogram('latentvar' + str(i), histogram, epoch.run.step)


if __name__ == '__main__':

    input_viewer = OpenCV('input', (320, 480))
    output_viewer = OpenCV('output', (320, 480))
    corr_viewer = OpenCV('correlation_matrix', (160, 160))


    def view_image(model, input, output):
        input_viewer.update(input[0][0].data)
        output_viewer.update(output[0][0].data)


    def store_latent_vars_in_epoch(model, input, output):
        if 'zl' not in model.run.epoch.context:
            model.run.epoch.context['zl'] = []
        model.run.epoch.context['zl'].append(output[0].data.cpu().numpy())

    dataset = torchvision.datasets.ImageFolder(
        root=config.datapath('spaceinvaders/images/raw'),
        transform=TVT.Compose([TVT.ToTensor()])
    )

    dataset = torchvision.datasets.ImageFolder(
        root=config.datapath('cartpole/images/raw'),
        transform=TVT.Compose([TVT.ToTensor()])
    )

    data_package = DataPackage(dataset, AutoEncodeSelect())

    run_fac = SimpleRunFac()
    #model = Params(ConvVAE4Fixed, (210, 160), 16)
    model = Params(ConvVAE4Fixed, (400, 600), 2)

    opt = Params(Adam, lr=1e-3)

    run_fac.run_list.append(Run(model, opt, Params(MseKldLoss), data_package, run_name='VAE loss'))
    run_fac.run_list.append(Run(model, opt, Params(MseKldLoss, beta=2.0), data_package, run_name='B-VAE loss 2.0'))
    #run_fac.run_list.append(Run(model, opt, Params(MseKldLoss, beta=4.0), data_package, run_name='B-VAE loss 4.0'))
    #run_fac.run_list.append(Run(model, opt, Params(MseKldLoss, beta=8.0), data_package, run_name='B-VAE loss 8.0'))

    #run_fac = SimpleRunFac.resume(r'C:\data\runs\417', data_package)

    batch_size = 10
    epochs = 10

    for model, opt, loss_fn, data_package, trainer, tester, run in run_fac:
        dev, train, test, selector = data_package.loaders(batch_size=batch_size)

        model.register_forward_hook(view_image)
        trainer.register_after_hook(tb_train_loss)
        trainer.register_after_hook(tb_image)
        tester.register_after_hook(tb_test_loss)
        tester.register_after_hook(tb_image)
        run.tb.add_graph(model, (data_package.dataset[0][0].cpu().unsqueeze(0),))

        for epoch in tqdm(run.for_epochs(epochs), 'epochs', epochs):

            epoch.register_after_hook(write_correlation)
            epoch.register_after_hook(write_histogram)
            epoch.execute_before(epoch)

            trainer.train(model, opt, loss_fn, train, selector, run, epoch)

            handles = Handles()
            handles += model.encoder.register_forward_hook(store_latent_vars_in_epoch)
            handles += loss_fn.register_hook(tb_train_loss_term)
            tester.test(model, loss_fn, test, selector, run, epoch)
            handles.remove()
            epoch.execute_after(epoch)
            run.save()









