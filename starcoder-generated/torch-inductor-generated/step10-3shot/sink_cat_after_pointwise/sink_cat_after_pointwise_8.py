
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2):
        k = x1.view(x1.shape[0], -1)
        k = k.relu().tanh()
        k = k.view(x1.shape[0], -1)
        return x2
# Inputs to the model
x1 = torch.randn(2, 3, 4)
x2 = torch.randn(2, 2, 4)
