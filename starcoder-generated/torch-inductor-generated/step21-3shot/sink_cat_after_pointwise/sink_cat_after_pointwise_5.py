
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        y = x.view(x.shape[0], -1).sigmoid()
        y0 = x.view(x.shape[0], -1).tanh()
        if y0.dim() == 1:
            y0 = y0.unsqueeze(1)
        return torch.cat((y0, y), dim=1)
# Inputs to the model
x = torch.randn(2, 3, 4)
