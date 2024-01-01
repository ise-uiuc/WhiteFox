
class ModelTanh(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(2, 2, kernel_size=(2, 2), stride=1, padding=0)
    def forward(self, x):
        v1 = self.conv1(x)
        y2 = torch.tanh(v1)
        return y2
# Inputs to the model
x = torch.randn(1, 2, 4, 5)
