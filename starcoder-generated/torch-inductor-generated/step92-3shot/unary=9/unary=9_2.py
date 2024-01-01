
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1):
        for _ in range(3):
            x1 = torch.Tensor.add(x1, 1)
        return x1
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
