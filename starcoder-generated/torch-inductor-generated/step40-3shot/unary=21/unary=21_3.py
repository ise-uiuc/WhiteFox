
class ModelTanh(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(6, 18, kernel_size=(1,7), stride=(1,1), padding=(0,3))
    def forward(self, x):
        v1 = self.conv1(x)
        return torch.tanh(v1)
# Inputs to the model
x = torch.randn(1, 6, 28, 160)
