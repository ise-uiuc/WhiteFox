
class Model2(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3,8,1,stride=1,padding=0)
        self.dropout1 = torch.nn.Dropout2d(p=0.5)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = self.dropout1(v1)
        return v2
# Inputs to the model
x1 = torch.randn(5, 3, 50, 50)
