
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        x = x * 0.5
        if x.dim() == 2:
            x = x.view(-1)
        return torch.cat((x, x), dim=0)
# Inputs to the model
x = torch.randn(2, 3, 4)
