
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.maxpool = torch.nn.MaxPool1d(2, stride=2)
        self.logsoftmax = torch.nn.LogSoftmax(dim=-1)
    def forward(self, x1):
        v1 = torch.cat([x1, x1], dim=0)
        v2 = self.maxpool(v1)
        v2 = self.maxpool(v1)
        v2 = self.maxpool(v1)
        v2 = self.maxpool(v1)
        v2 = self.maxpool(v1)
        v2 = self.maxpool(v1)
        v2 = self.maxpool(v1)
        v2 = self.maxpool(v1)
        v2 = self.maxpool(v1)
        v2 = self.logsoftmax(v1)
        v2 = self.logsoftmax(v2)
        v2 = self.logsoftmax(v2)
        return v2
# Inputs to the model
x1 = torch.randn(1, 1, 2)
