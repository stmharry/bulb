import abc
import torch

from bulb.saver import Saver


class Net(object):
    __metaclass__ = abc.ABCMeta

    def __init__(self,
                 model,
                 data_loader=None,
                 writer=None,
                 **kwargs):

        self.model = model
        self.writer = writer
        self.data_loader = self.data_loader

        self.num_epoch = 0

        self.init(**kwargs)

    def init(self):
        pass

    def _register_vars(self, vars_dict):
        for (var_name, var) in vars_dict.items():
            setattr(self, var_name, var)

    def pre_batch(self):
        pass

    @abc.abstractmethod
    def step_batch(self):
        pass

    def post_batch(self):
        pass

    def log(self):
        strings = [
            '[{:s}]'.format(self.name),
            'epoch: {:d}'.format(self.num_epoch + 1),
            'batch: {:d}/{:d}'.format(self.num_batch + 1, len(self.data_loader)),
            'loss: {:.4f}'.format(self.loss.item()),
        ]

        print('\t'.join(strings))

    def pre_epoch(self):
        pass

    def step_epoch(self, num_step=None):
        self.pre_epoch()

        for (self.num_batch, vars_dict) in enumerate(self.data_loader):
            self.num_step = num_step or self.num_epoch * len(self.data_loader) + self.num_batch

            self._var_names = vars_dict.keys()
            self._register_vars(vars_dict)

            self.pre_batch()

            losses_dict = self.step_batch()
            losses_dict['total_loss'] = sum(losses_dict.values())

            self._loss_names = losses_dict.keys()
            self._register_vars(losses_dict)

            self.post_batch()
            self.log()

        self.post_epoch()

        self.num_epoch += 1

    def post_epoch(self):
        pass


class TrainMixin(object):
    name = 'train'

    def init(self,
             lr,
             lr_decay_epochs,
             lr_decay_rate,
             summarize_steps,
             save_steps,
             saver):

        self.lr = lr
        self.lr_decay_epochs = lr_decay_epochs
        self.lr_decay_rate = lr_decay_rate
        self.summarize_steps = summarize_steps
        self.save_steps = save_steps
        self.saver = saver

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        if lr_decay_epochs is None:
            self.scheduler = None
        else:
            self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=self.lr_decay_epochs, gamma=self.lr_decay_rate)

    def summarize(self):
        for loss_name in self._loss_names:
            loss = getattr(self, loss_name)

            if self.writer is not None:
                self.writer.add_scalar(
                    '{:s}/{:s}'.format(self.name, loss_name),
                    loss.item(),
                    self.num_step,
                )

    def save(self):
        state_dict = {
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }
        self.saver.save_model(state_dict, num_step=self.num_step)

    def load(self, ckpt_dir=None):
        state_dict = Saver.load_model(ckpt_dir=ckpt_dir)
        self.model.load_state_dict(state_dict['model'])
        self.optimizer.load_state_dict(state_dict['optimizer'])

    def pre_epoch(self):
        self.model.train()

    def pre_batch(self):
        for var_name in self._var_names:
            var = getattr(self, var_name)
            if not isinstance(var, list):
                var = torch.Tensor(var, requires_grad=True).cuda()
                setattr(self, var_name, var)

    def post_batch(self):
        self.optimizer.zero_grad()
        self.loss.backward()
        self.optimizer.step()

        if (self.num_step % self.summarize_steps == 0):
            self.summarize()

        if (self.num_step % self.save_steps == 0):
            self.save()

    def post_epoch(self):
        if self.scheduler is not None:
            self.scheduler.step(epoch=self.num_epoch)

        if self.writer is not None:
            self.writer.add_scalar(
                '{:s}/lr'.format(self.name),
                self.optimizer.param_groups[0]['lr'],
                self.num_step,
            )


class TestMixin(object):
    name = 'test'

    def summarize(self):
        for loss_name in self._loss_names:
            loss = self._cum_losses[loss_name] / len(self.data_loader)

            if self.writer is not None:
                self.writer.add_scalar(
                    '{:s}/{:s}'.format(self.name, loss_name),
                    loss,
                    self.num_step,
                )

    def load(self, ckpt_dir=None):
        state_dict = Saver.load_model(ckpt_dir=ckpt_dir)
        self.model.load_state_dict(state_dict['model'])

    def pre_epoch(self):
        self.model.eval()
        self._cum_losses = {}

    def pre_batch(self):
        for var_name in self._var_names:
            var = getattr(self, var_name)
            if not isinstance(var, list):
                var = torch.Tensor(var).cuda()
                setattr(self, var_name, var)

    def post_batch(self):
        for loss_name in self._loss_names:
            if loss_name in self._cum_losses:
                _cum_loss = self._cum_losses[loss_name]
            else:
                _cum_loss = 0.

            loss = getattr(self, loss_name)
            self._cum_losses[loss_name] = _cum_loss + loss.item()

    def post_epoch(self):
        self.summarize()
