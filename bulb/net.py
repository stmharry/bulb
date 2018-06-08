import torch

from bulb.saver import Saver


class Net(object):
    def __init__(self,
                 model,
                 data_loader=None,
                 writer=None,
                 **kwargs):

        self.model = model
        self.writer = writer
        self.data_loader = data_loader

        self.num_epoch = 0

        self._init(**kwargs)

    def _register_vars(self, vars_dict):
        for (var_name, var) in vars_dict.items():
            setattr(self, var_name, var)

    def _prepare(self, requires_grad):
        for var_name in self._var_names:
            var = getattr(self, var_name)
            var = torch.tensor(var, requires_grad=requires_grad).cuda()
            setattr(self, var_name, var)

    def _process(self):
        for var_name in self._loss_names + self._metric_names:
            var = getattr(self, var_name)
            setattr(self, var_name, var.item())

    def _log(self):
        strings = [
            '[{:s}]'.format(self.name),
            'epoch: {:d}'.format(self.num_epoch + 1),
            'batch: {:d}/{:d}'.format(self.num_batch + 1, len(self.data_loader)),
            'loss: {:.4f}'.format(self.loss),
        ]

        print('\t'.join(strings))

    def _summarize(self):
        for var_name in self._loss_names + self._metric_names:
            var = getattr(self, var_name)

            if self.writer is not None:
                self.writer.add_scalar(
                    '{:s}/{:s}'.format(self.name, var_name),
                    var,
                    self.num_step,
                )

    def step_epoch(self, num_step=None):
        self.pre_epoch()

        for (self.num_batch, vars_dict) in enumerate(self.data_loader):
            self.num_step = num_step or self.num_epoch * len(self.data_loader) + self.num_batch

            self._var_names = vars_dict.keys()
            self._register_vars(vars_dict)

            self.pre_batch()

            result = self.step_batch()
            losses_dict = result['loss']
            metrics_dict = result['metrics']

            losses_dict['loss'] = sum(losses_dict.values())

            self._loss_names = losses_dict.keys()
            self._register_vars(losses_dict)

            self._metric_names = metrics_dict.keys()
            self._register_vars(metrics_dict)

            self.post_batch()

        self.post_epoch()

        self.num_epoch += 1


class TrainMixin(object):
    name = 'train'

    def _init(self,
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

    def _optimize(self):
        self.optimizer.zero_grad()
        self.loss.backward()
        self.optimizer.step()

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

    def pre_batch(self):
        self._prepare(requires_grad=True)

    def post_batch(self):
        self._optimize()
        self._process()

        self._log()

        if (self.num_step % self.summarize_steps == 0):
            self._summarize()

        if (self.save_steps is not None) and (self.num_step % self.save_steps == 0):
            self.save()

    def pre_epoch(self):
        self.model.train()

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

    def _init(self):
        pass

    def load(self, ckpt_dir=None):
        state_dict = Saver.load_model(ckpt_dir=ckpt_dir)
        self.model.load_state_dict(state_dict['model'])

    def pre_batch(self):
        self._prepare(requires_grad=False)

    def post_batch(self):
        self._process()

        for var_name in self._loss_names + self._metric_names:
            if var_name in self._loss_metrics:
                _var = self._loss_metrics[var_name]
            else:
                _var = 0

            var = getattr(self, var_name)
            self._loss_metrics[var_name] = _var + (var - _var) / (self.num_batch + 1)

        self._log()

    def pre_epoch(self):
        self.model.eval()
        self._loss_metrics = {}

    def post_epoch(self):
        for var_name in self._loss_names + self._metric_names:
            setattr(self, var_name, self._loss_metrics[var_name])

        self._summarize()
