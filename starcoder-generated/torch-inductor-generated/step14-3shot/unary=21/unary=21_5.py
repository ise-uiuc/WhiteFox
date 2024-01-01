
class ModelTanh(torch.nn.Module):
    def __init__(self):
        super(ModelTanh, self).__init__()
        self.conv = torch.nn.Conv2d(259, 128, (6, 1), stride=(48, 1), bias=True)
    def forward(self, x):
        v1 = self.conv(x)
        v2 = torch.tanh(v1)
        return v2
# Input tensor to the model
x = torch.randn(1, 259, 20, 2)
