
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        y = x.view(-1)
        if y.dim() > 1:
            return y
        return y.reshape(1, 2, -1)
# Inputs to the model
x = torch.randn(4)
