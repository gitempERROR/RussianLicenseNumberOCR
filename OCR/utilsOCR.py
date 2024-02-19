import torch
from OCR import configOCR
import string
import numpy as np
from PIL import Image


def save_model_checkpoint(model, optimizer, checkpoint_path):
    print('Saving checkpoint...', end='')
    checkpoint = {
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict()
    }
    torch.save(checkpoint, checkpoint_path)
    print('  Save complete!')


def load_model_checkpoint(model, optimizer):
    print('Loading checkpoint...', end='')
    checkpoint = torch.load(f=configOCR.MODEL_DIR, map_location=configOCR.DEVICE)
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])

    for param_group in optimizer.param_groups:
        param_group['lr'] = configOCR.LEARNING_RATE

    print('  Load complete!')


def print_result(predictions):
    symbol_list = configOCR.LETTER_LIST + list(string.digits)
    predictions = torch.argmax(predictions.to('cpu'), -1)
    for symbol in predictions[0]:
        print(symbol_list[symbol], end='')
    print()


def test_func(model):
    img = np.array(Image.open(r"C:\Programming\Datasets\License Plate OCR\O853BC68.png").convert('RGB'))
    img = configOCR.TRANSFORMS(image=img)['image']
    img = img.to(configOCR.DEVICE).unsqueeze(0)
    with torch.no_grad():
        predictions = model(img)
    predictions = torch.argmax(predictions.to('cpu'), -1)

    symbol_list = configOCR.LETTER_LIST + list(string.digits)
    for symbol in predictions[0]:
        print(symbol_list[symbol], end='')
    print()