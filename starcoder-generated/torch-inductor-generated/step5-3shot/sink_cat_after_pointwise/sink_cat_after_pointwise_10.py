
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        y = x.view(x.shape[0], -1).numel()
        y = torch.full([y], 1)
        if y[0] > 0:
            x = x.view(x.shape[0], -1).tanh()
        else:
            x = torch.full((y if y!= 0 else 1,), 2.3).tanh()
        return x
# Inputs to the model
x = torch.randn(2, 3, 4)
