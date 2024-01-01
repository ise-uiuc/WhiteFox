
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.block1 = torch.nn.Sequential(torch.nn.Conv2d(3, 3, 3, padding=1), torch.nn.BatchNorm2d(3, affine=True))
        self.block2 = torch.nn.Sequential(torch.nn.Conv2d(3, 3, 3), torch.nn.BatchNorm2d(3, affine=False))
        torch.manual_seed(2)
        self.dropout = torch.nn.Dropout(p=0.4)
        self.relu = torch.nn.ReLU()
        torch.manual_seed(3)
        self.pooling = torch.nn.AvgPool2d(3, ceil_mode=True)
    def forward(self, x1):
        s1 = self.block1(x1)
        s2 = self.block2(s1)
        s2 = self.dropout(s2)
        s2 = self.relu(s2)
        s2 = self.pooling(s2)
        return s2
# Inputs to the model
x1 = torch.randn(1, 3, 4, 4)
