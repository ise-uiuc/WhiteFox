
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        y = x.view(-1)
        w = y.reshape(-1)
# Inputs to the model
x = torch.randn(2,)
