
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        xa = x.view(3, 3)
        xb = xa.flip(dims=([0, 1], [1, 0]))
        x = torch.relu(xb)
        return x
# Inputs to the model
x = torch.randn(3, 4, 5)
