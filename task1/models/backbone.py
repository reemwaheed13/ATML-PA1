import torch
import torch.nn as nn
import torchvision.models as models
import open_clip


RESNET50_DIM = 2048
VIT_B16_DIM = 768
CLIP_VIT_B32_DIM = 512


class ResNet50Backbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        self.backbone.fc = nn.Identity()  # drop the 1000-class head; output is 2048-d pool
        for param in self.backbone.parameters():
            param.requires_grad = False
        self.backbone.eval()  # use pretrained BN running stats, not batch stats

    def train(self, mode=True):
        # backbone is always frozen — keep it in eval regardless of outer .train() calls
        return self

    def forward(self, x):
        return self.backbone(x)


class ViTBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.vit = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1)
        self.vit.heads = nn.Identity()  # drop classification head; output is 768-d CLS token
        for param in self.vit.parameters():
            param.requires_grad = False
        self.vit.eval()

    def train(self, mode=True):
        return self

    def forward(self, x):
        return self.vit(x)


class CLIPBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        # preprocess is CLIP's val transform (resize + center-crop + CLIP-specific normalization).
        # Store it so dataloaders can apply the right normalization for this backbone.
        self.clip, _, self.preprocess = open_clip.create_model_and_transforms(
            'ViT-B-32', pretrained='openai'
        )
        for param in self.clip.parameters():
            param.requires_grad = False
        self.clip.eval()

    def train(self, mode=True):
        return self

    def forward(self, x):
        features = self.clip.encode_image(x)
        return features / features.norm(dim=-1, keepdim=True)  # L2-normalize to unit sphere


class LinearHead(nn.Module):
    def __init__(self, in_features, num_classes=10):
        super().__init__()
        self.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.fc(x)


def zero_shot_clip_predict(clip_backbone, images, class_names):
    """
    Classify images using CLIP's text-image similarity, no training required.
    clip_backbone: a CLIPBackbone instance.
    images: preprocessed image tensor (already passed through clip_backbone.preprocess).
    class_names: list of string class names, e.g. ['airplane', 'bird', ...].
    Returns: predicted class indices (LongTensor of shape [N]).
    """
    model = clip_backbone.clip
    tokenizer = open_clip.get_tokenizer('ViT-B-32')
    prompts = [f"a photo of a {c}" for c in class_names]
    text_tokens = tokenizer(prompts).to(images.device)
    with torch.no_grad():
        text_features = model.encode_text(text_tokens)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        image_features = clip_backbone(images)  # already L2-normalized by forward()
    logits = image_features @ text_features.T  # [N, num_classes]
    return logits.argmax(dim=-1)
