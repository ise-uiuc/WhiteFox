
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        return x.tanh()
# Inputs to the model
x = torch.randn(10)
