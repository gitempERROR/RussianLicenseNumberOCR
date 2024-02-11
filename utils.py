import torch
import configOCR


def save_model_checkpoint(model, optimizer, checkpoint_path):
    print('Saving checkpoint...', end='')
    checkpoint = {
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict()
    }
    torch.save(checkpoint_path, checkpoint)
    print('  Save complete!')


def load_model_checkpoint(model, optimizer, checkpoint_path):
    print('Loading checkpoint...', end='')
    checkpoint = torch.load(f=checkpoint_path, map_location=configOCR.DEVICE)
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])

    for param_group in optimizer.param_groups:
        param_group['lr'] = configOCR.LEARNING_RATE

    print('  Load complete!')
