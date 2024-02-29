import clip
import torch.nn as nn

class TextCLIP(nn.Module):
    def __init__(self, model):
        super(TextCLIP, self).__init__()
        self.model = model

    def forward(self, text):
        return self.model.encode_text(text)


class ImageCLIP(nn.Module):
    def __init__(self, model):
        super(ImageCLIP, self).__init__()
        self.model = model

    def forward(self, image, prompt_token=None, local_ind_prompt=None):
        # return self.model.encode_image(image, prompt_token, local_ind_prompt=local_ind_prompt)
        return self.model.encode_image(image)


class ParallCLIP(nn.Module):
    def __init__(self, model, device='cuda'):
        super(ParallCLIP, self).__init__()

        self.model = nn.DataParallel(model)
        self.model.to(device)

    def forward(self, input, modality='text'):
        if modality == 'text':
            return self.model.module.encode_text(input)
        elif modality == 'image':
            return self.model.module.encode_image(input)

    def encode_text(self, input):
        return self.forward(input, 'text')

    def encode_image(self, input):
        return self.forward(input, 'image')


def build_clip(model_size: str, device: str, jit: bool, parallel: bool):
    # load model
    encoder, preprocesser = clip.load(model_size, device=device, jit=jit)
    encoder.eval()
    encoder.requires_grad_(False)

    if parallel:
        return ParallCLIP(encoder, device=device), preprocesser
    else:
        return encoder, preprocesser

