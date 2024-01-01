
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        v = x.new_ones(1)
        v = torch.nn.functional.dropout2d(x, p=0.5, train=True, inplace=False)
        x2 = v + x
        return torch.nn.functional.dropout2d(x2, p=0.5, train=True, inplace=False)

# Inputs to the model
x1 = torch.randn(1, 2, 2, 2)
