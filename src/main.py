import torch

import utility
import data
import model
from loss import first_stage_loss
from option import args
from loss import second_stage_loss, third_stage_loss
from trainer import Trainer
import logging

torch.manual_seed(args.seed)
checkpoint = utility.checkpoint(args)


def train():
    logging.info("no down sample start")
    logging.info(args)
    if checkpoint.ok:
        loader = data.Data(args)
        _model = model.Model(args, checkpoint)
        total_params = sum(p.numel() for p in _model.parameters())
        print(f'{total_params:,} total parameters.')
        _first_stage_loss = first_stage_loss.First_Stage_Loss(args, checkpoint) if not args.test_only else None
        _second_stage_loss = second_stage_loss.Second_Stage_Loss(args, checkpoint) if not args.test_only else None
        _third_stage_loss = third_stage_loss.Third_Stage_Loss(args, checkpoint) if not args.test_only else None
        losses = [_first_stage_loss, _second_stage_loss, _third_stage_loss]
        t = Trainer(args, loader, _model, losses, checkpoint)
        while not t.terminate():
            t.train()
            t.test()
        checkpoint.done()


if __name__ == '__main__':
    LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
    DATE_FORMAT = "%m/%d/%Y %H:%M:%S %p"
    logging.basicConfig(filename='my.log', level=logging.DEBUG, format=LOG_FORMAT, datefmt=DATE_FORMAT)
    train()
