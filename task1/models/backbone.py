import torch
import torch.nn as nn
import torchvision.models as models
import open_clip

class ResNet50Backbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        self.backbone.fc = nn.Identity()
        for param in self.backbone.parameters():
            param.requires_grad = False

    def forward(self, x):
        return self.backbone(x)

class ViTBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.vit = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1)
        self.vit.heads = nn.Identity() #now we just output the cls tok
        for param in self.vit.parameters():
            param.requires_grad = False

    def forward(self, x):
        return self.vit(x)


class CLIPBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.clip, _, _ = open_clip.create_model_and_transforms('ViT-B-32', pretrained='openai')
        for param in self.clip.parameters():
            param.requires_grad = False

    def forward(self, x):
        #extract features from clip
        features = self.clip.encode_image(x)
        #normalize them
        return features / features.norm(dim=-1, keepdim=True) #normalize along the 512 features dimension


class LinearHead(nn.Module):
    def __init__(self, in_features, num_classes=10):
        super().__init__()
        # stl always has 10
        self.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.fc(x)



def zero_shot_clip_predict(clip_model, images, class_names):
    model = clip_model.clip if hasattr(clip_model, 'clip') else clip_model
    prompts = [f"a photo of a {c}" for c in class_names]
    tokenizer = open_clip.get_tokenizer('ViT-B-32')
    text_tokens = tokenizer(prompts)
    text_features = model.encode_text(text_tokens)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    image_features = clip_model(images) if callable(clip_model) else model.encode_image(images)
    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    logits = image_features @ text_features.T
    return logits.argmax(dim=-1)
