
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.maxPool = torch.nn.MaxPool3d(kernel_size=(3, 3, 3), stride=(2, 2, 2), padding=0)
        self.reLu = torch.nn.ReLU()
    def forward(self, x):
        v1 = self.maxPool(x)
        v2 = self.reLu(v1)
        return v2
# Inputs to the model
x = torch.randn(3, 16, 32, 32, 32)
