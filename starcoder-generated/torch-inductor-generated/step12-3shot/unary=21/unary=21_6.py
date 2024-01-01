
class ModelTanh(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 1, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
    def forward(self, x):
        t1 = self.conv1(x)
        t2 = torch.tanh(t1)
        return t2
# Inputs to the model
x = torch.randn(1, 3, 128, 128)
