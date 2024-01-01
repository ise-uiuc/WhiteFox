
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.bn1 = torch.nn.BatchNorm1d(6)
        self.bn2 = torch.nn.BatchNorm1d(6)
        self.bn3 = torch.nn.BatchNorm1d(6)
        self.bn4 = torch.nn.BatchNorm1d(6)
    def forward(self,x3):
        s1 = self.bn1(x3)
        s2 = self.bn2(s1)
        s3 = self.bn3(s2)
        s4 = self.bn4(s3)
        return s4[0]
# Inputs to the model
x3 = torch.randn(3, 6, requires_grad=False)
