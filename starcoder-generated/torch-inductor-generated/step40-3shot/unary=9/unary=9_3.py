
class Model(nn.Module):
    def __init__(self):
        super().__init__()        
        self.features = nn.Sequential()
        self.features.add_module('conv1', nn.Conv2d(3, 8, 9))
    def forward(self, x1):
        x1 = F.conv2d(x1, self.features[0].weight, stride=(1, 1), padding=(4, 4))
        return x1
# Inputs to the model
x1 = torch.randn(3, 3, 96, 96)
