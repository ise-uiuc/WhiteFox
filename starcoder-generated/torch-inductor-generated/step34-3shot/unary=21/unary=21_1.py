
class ModelTanh(torch.nn.Module):
    def __init__(self):
        super().__init__(3, 8, kernel_size=3, padding=0, bias=True)
    def forward(self, x):
        v = torch.tanh(super().forward(x))
        return v
# Inputs to the model
x = torch.randn(4, 3, 512, 1024)
