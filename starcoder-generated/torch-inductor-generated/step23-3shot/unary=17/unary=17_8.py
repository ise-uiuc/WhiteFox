
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1):
        v1 = x1.flip(2)
        return v1
# Inputs to the model
x1 = torch.randn(1, 3, 128, 128)
