import numpy as np

from mentalitystorm import Handles, OpenCV

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
        input_image = args.input_data[0][0, 0:3].data
        output_image = args.output_data[0][0, 0:3].data
        args.run.tb.add_image('input', input_image, args.run.step)
        args.run.tb.add_image('output', output_image, args.run.step)




def write_histogram(epoch):
    z = np.concatenate(epoch.context['zl'], axis=0).squeeze()
    histograms = np.rollaxis(z, 1)
    for i, histogram in enumerate(histograms):
        epoch.run.tb.add_histogram('latentvar' + str(i), histogram, epoch.run.step)




def register_tb(run):
    handles = Handles()
    handles += run.trainer.register_after_hook(tb_train_loss)
    handles += run.trainer.register_after_hook(tb_image)
    handles += run.tester.register_after_hook(tb_test_loss)
    handles += run.tester.register_after_hook(tb_image)
    handles += run.loss.register_hook(tb_train_loss_term)
    run.tb.add_graph(run.model, (run.data_package.dataset[0][0].cpu().unsqueeze(0),))
    return handles


class LatentInstrument:
    def __init__(self):
        self.corr_viewer = OpenCV('correlation_matrix', (160, 160))

    def store_latent_vars_in_epoch(self, model, input, output):
        if 'zl' not in model.run.epoch.context:
            model.run.epoch.context['zl'] = []
        model.run.epoch.context['zl'].append(output[0].data.cpu().numpy())

    def write_correlation(self, epoch):
        z = np.concatenate(epoch.context['zl'], axis=0).squeeze()
        corr = np.corrcoef(z, rowvar=False)
        corr = np.absolute(corr)
        epoch.run.tb.add_scalar('z/z_ave_correlation', (corr - np.identity(corr.shape[0])).mean(), epoch.run.step)
        corr = np.expand_dims(corr, axis=0)
        self.corr_viewer.update(corr)
        epoch.run.tb.add_image('corr_matrix', corr, epoch.run.step)
