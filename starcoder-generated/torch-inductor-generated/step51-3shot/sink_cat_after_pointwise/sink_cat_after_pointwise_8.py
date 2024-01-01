
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        if x.shape[1] > 3:
            return x.view(-1, 1)
        else:
            return x.view(-1, 1, x.shape[2]).repeat(1, 2, 1)
# Inputs to the model
x = torch.randn(2, 3, 4)
