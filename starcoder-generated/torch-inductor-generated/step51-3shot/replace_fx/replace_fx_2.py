
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(...)
        self.conv2 = nn.Conv2d(...)
        self.conv3 = nn.ConvTranspose2d(...)
        
        self.conv4 = nn.Conv2d(...)
        self.conv5 = nn.Conv2d(...)
        self.bn1 = nn.BatchNorm2d(...)
    def forward(self, x1):
        x2 = self.conv1(x1)
        x2 = self.conv2(x2)
        x2 = self.conv3(x2)
        
        x3 = self.conv4(x1)
        x3 = self.conv5(x3)
        x3 = self.bn1(x3)
        
        x = torch.rand_like(x2)
        x = torch.nn.functional.dropout(x, p=0.3)
        return x
# Inputs to the model
x1 = torch.randn(4, 3, 10, 10)
