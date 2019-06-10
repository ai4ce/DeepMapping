import os
import json
import torch


def save_opt(working_dir, opt):
    """
    Save option as a json file
    """
    opt = vars(opt)
    save_name = os.path.join(working_dir, 'opt.json')
    with open(save_name, 'wt') as f:
        json.dump(opt, f, indent=4, sort_keys=True)


def save_checkpoint(save_name, model, optimizer):
    state = {'state_dict': model.state_dict(),
             'optimizer': optimizer.state_dict()}
    torch.save(state, save_name)
    print('model saved to {}'.format(save_name))


def load_checkpoint(save_name, model, optimizer):
    state = torch.load(save_name)
    model.load_state_dict(state['state_dict'])
    if optimizer is not None:
        optimizer.load_state_dict(state['optimizer'])
    print('model loaded from {}'.format(save_name))


def load_opt_from_json(file_name):
    if os.path.isfile(file_name):
        with open(file_name,'rb') as f:
            opt_dict = json.load(f)
            return opt_dict
    else:
        raise FileNotFoundError("Can't find file: {}. Run training script first".format(file_name))
