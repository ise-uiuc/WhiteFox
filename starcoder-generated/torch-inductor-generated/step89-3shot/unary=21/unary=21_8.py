
class ModelTanh(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv2d = None
    def forward(self, x):
        v1 = self.conv2d(x)
        v2 = torch.tanh(v1)
        return v2
# Inputs to the model
x = torch.zeros(1, 3, 256, 256)
