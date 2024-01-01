
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        y = torch.cat((x, x), dim=1)
        if y.dim() < 2 or y.size(0) < 2:
            return y.unsqueeze(0).expand(2, y.size(0), y.size(1)).tanh()
        else:
            return y.unsqueeze(0).expand(2, y.size(0), y.size(1)).tanh()
# Inputs to the model
x = torch.randn(1, 2)
