
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dropout = torch.nn.Dropout2d(p=0.01)
        self.conv = torch.nn.Conv2d(1, 1, 3, 1, 1)
    def forward(self, x1):
        v1 = self.dropout(x1)
        v2 = self.conv(v1)
        return nn.Sigmoid()(v2)
# Inputs to the model
x1 = torch.randn(1, 1, 224, 224)
