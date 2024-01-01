
class ModelTanh(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = torch.nn.Conv2d(3, 16, kernel_size=(5,5), padding=(4, 4))
    def forward(self, x):
        v1 = self.l1(x)
        v2 = torch.tanh(v1)
        return v2
# Inputs to the model
x = torch.randn(1, 3, 224, 224)
