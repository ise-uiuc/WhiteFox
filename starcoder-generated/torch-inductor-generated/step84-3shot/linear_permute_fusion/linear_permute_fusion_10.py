
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        return x.view(x.shape[1])
# Inputs to the model
x1 = torch.randn(3, 2, 2)
