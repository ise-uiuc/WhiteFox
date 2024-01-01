
class ModelTanh(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = None
    def forward(self, x):
        v1 = x.view(x.shape[0], -1)
        v3 = torch.tanh(v1)
        return v1, v3
# Inputs to the model
x = torch.randn(10, 3, 28, 28)
