import torch
import torch.nn as nn
import torchvision.models as models

class DenseNetWrapper(nn.Module):
    def __init__(self):
        super(DenseNetWrapper, self).__init__()
        self.base_model = models.regnet_y_1_6gf(pretrained=True)
        self.stem=self.base_model.stem
        self.features=self.base_model.trunk_output

    def forward(self, x):
        layers = [0,1,2,3]  # 这里选择第4、第7、第14和第18个层
        outputs = []
        x=self.stem(x)
        for idx, module in enumerate(self.features):
            x = module(x)
            if idx in layers:
                outputs.append(x)
        return outputs  
    
