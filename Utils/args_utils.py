import logging
import time
import os
import sys
from Utils.logger import create_exp_dir


class ArgsInit(object):
    def __init__(self, parser):
        self.args = parser.parse_args()

    def save_exp(self):
        self.args.save = 'D_{}-L_{}-J_{}-E_{}-B_{}'.format(self.args.save + "_",
                                                                             self.args.dataset,
                                                                             self.args.num_layer, self.args.JK,
                                                                             self.args.epochs, self.args.batch_size)

        self.args.save = 'log/{}-{}'.format(self.args.save, time.strftime("%Y%m%d-%H%M%S"))
        if not self.args.model_save_path == '':
            self.args.model_save_path = os.path.join(self.args.save, self.args.model_save_path)
            create_exp_dir(self.args.save)
            fh = logging.FileHandler(os.path.join(self.args.save, 'log.txt'))
        else:
            fh = logging.FileHandler('{}-{}'.format(self.args.save, '.txt'))

        log_format = '%(asctime)s %(message)s'
        logging.basicConfig(stream=sys.stdout,
                            level=logging.INFO,
                            format=log_format,
                            datefmt='%m/%d %I:%M:%S %p')
        fh.setFormatter(logging.Formatter(log_format))
        log = logging.getLogger()
        log.addHandler(fh)

        return self.args

    def save_exp4pretrain(self):
        self.args.save = '{}-D_{}-L_{}-LR_{}-J_{}-E_{}-B_{}'.format(self.args.save, self.args.dataset,
                                                                             self.args.num_layer, self.args.lr, self.args.JK,
                                                                             self.args.epochs, self.args.batch_size)

        self.args.save = 'log/{}'.format(self.args.save)
        if not self.args.model_save_path == '':
            self.args.model_save_path = os.path.join(self.args.save, self.args.model_save_path)
            create_exp_dir(self.args.save)
            fh = logging.FileHandler(os.path.join(self.args.save, 'log.txt'))
        else:
            fh = logging.FileHandler('{}-{}'.format(self.args.save, '.txt'))

        log_format = '%(asctime)s %(message)s'
        logging.basicConfig(stream=sys.stdout,
                            level=logging.INFO,
                            format=log_format,
                            datefmt='%m/%d %I:%M:%S %p')
        fh.setFormatter(logging.Formatter(log_format))
        log = logging.getLogger()
        log.addHandler(fh)

        return self.args
