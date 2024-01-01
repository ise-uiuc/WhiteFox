
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        y = x.view(x.shape[0], -1).sum(dim=0).unsqueeze(dim=0)
        return x + y
# Inputs to the model
x = torch.randn(2, 2, 2)
