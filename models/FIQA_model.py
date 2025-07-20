import torch
import torch.nn as nn
import timm
import torchvision.models as models

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
        
    def forward(self, x):
        return x

class Swin_b_IQA(nn.Module):
    def __init__(self, is_pretrained=False):
        super(Swin_b_IQA, self).__init__()

        if is_pretrained:
            model = models.swin_b(weights='Swin_B_Weights.DEFAULT')
        else:
            model = models.swin_b()

        model.head = Identity()
        self.feature_extraction = model

        self.quality = self.quality_regression(1024, 128, 1)

    def quality_regression(self,in_channels, middle_channels, out_channels):
        regression_block = nn.Sequential(
            nn.Linear(in_channels, middle_channels),
            nn.Linear(middle_channels, out_channels),          
        )

        return regression_block

    def forward(self, x):
            
        x = self.feature_extraction(x)
        x = self.quality(x)
                    
        return x

class FIQA_Swin_B(torch.nn.Module):
    # pretrained_path: use the pretrained model on IQA/VQA datasets to initialize the model
    def __init__(self, pretrained_path):
        
        super(FIQA_Swin_B, self).__init__()

        swin_b = Swin_b_IQA()
        if pretrained_path!=None:
            print('load overall model')
            swin_b.load_state_dict(torch.load(pretrained_path))
        swin_b.quality = Identity()

        self.feature_extraction = swin_b

        self.quality = self.quality_regression(1024, 128, 1)

    def quality_regression(self,in_channels, middle_channels, out_channels):
        regression_block = nn.Sequential(
            nn.Linear(in_channels, middle_channels),        
            nn.Linear(middle_channels, out_channels),          
        )

        return regression_block

    def forward(self, x):

        x = self.feature_extraction(x)

        x = self.quality(x)
            
        return x

class FIQA_EdgeNeXt_XXS(nn.Module):
    def __init__(self, is_pretrained=True):
        super(FIQA_EdgeNeXt_XXS, self).__init__()

        if is_pretrained:
            model = timm.create_model('edgenext_xx_small', pretrained=True)
        else:
            model = timm.create_model('edgenext_xx_small', pretrained=False)
        
        model.head = Identity()

        self.feature_extraction = model
        self.avg_pool = torch.nn.AdaptiveAvgPool2d((1,1))

        self.quality = self.quality_regression(168, 128, 1)

    def quality_regression(self, in_channels, middle_channels, out_channels):
        regression_block = nn.Sequential(
            nn.Linear(in_channels, middle_channels),
            nn.Linear(middle_channels, out_channels),
        )
        return regression_block

    def forward(self, x):
        x = self.feature_extraction(x)
        x = self.avg_pool(x)
        x = torch.flatten(x, 1)
        x = self.quality(x)
        return x