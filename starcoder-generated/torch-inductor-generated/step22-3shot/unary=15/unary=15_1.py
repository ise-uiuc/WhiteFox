
class Model(torch.nn.Module):    
    def __init__(self):
        super(Model, self).__init__()
        self._modules = {'conv1':torch.nn.Conv2d(3, 24, (1, 1), stride=(1, 1), bias=False),
                        'conv2':torch.nn.Conv2d(24, 16, (1, 1), stride=(1, 1), bias=False),
                        'conv3':torch.nn.Conv2d(16, 8, (3, 4), stride=(1, 1), bias=False),
                      'relu':torch.nn.ReLU()}
    def forward(self, x1):
        x2 = self._modules['conv1'](x1)
        x3 = self._modules['conv2'](x2)
        x4 = self._modules['conv3'](x3)
        x5 = self._modules['relu'](x4)
        return x5
# Inputs to the model
x1 = torch.randn(1, 3, 32, 32)
