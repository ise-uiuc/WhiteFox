
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv1 = torch.nn.Conv2d(14, 64, (1, 1), stride=1, padding=0)
        self.conv2 = torch.nn.Conv2d(64, 64, (1, 1), stride=1, padding=0)
        self.conv3 = torch.nn.Conv2d(64, 1, (1, 1), stride=1, padding=0)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = torch.transpose(x, 1, 2).contiguous()
        x3 = self.conv2(x2)    
        x4 = torch.transpose(x3, 1, 2).contiguous()
        x5 = self.conv3(x4)
        return x5
# Inputs to the model
x1 = torch.randn(1, 14, 20, 20)
