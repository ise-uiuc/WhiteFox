
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x: torch.Tensor):
        y = x.tanh()
        y = torch.cat((y, y), dim=-1)
        x = x if torch.numel(x) == 1 else torch.cat((y, y), dim=-1).view(y.shape[0], -1)
        x.tanh()
        return x
# Inputs to the model
x = torch.randn(2, 2, 2)
