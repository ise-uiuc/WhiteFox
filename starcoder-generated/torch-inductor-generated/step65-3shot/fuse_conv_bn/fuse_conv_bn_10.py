
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(7)
        self.layer = torch.nn.Sequential(torch.nn.Conv3d(12, 18, 3, bias=True), torch.nn.ReLU6())
        torch.manual_seed(7)
        self.layer2 = torch.nn.Sequential(torch.nn.BatchNorm3d(18), torch.nn.Conv3d(18, 7, 5), torch.nn.ReLU6())
        torch.manual_seed(7)
        self.layer3 = torch.nn.Sequential(torch.nn.Conv3d(7, 6, 5, bias=False), torch.nn.BatchNorm3d(6), torch.nn.ReLU6())
    def forward(self, x1):
        s1 = self.layer(x1)
        s2 = self.layer2(s1)
        s3 = self.layer2(s2)
        s5 = self.layer3(s3)
        s7 = self.layer2(s5)
        return torch.cat((s7, s7), dim=1)
# Inputs to the model
x1 = torch.randn(1, 12, 28, 24, 18)
