
class model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        x = torch.cat([x, x], dim=0)
        x = x.view(3, -1)
        x = x.permute(1, 0)
        return x
# Inputs to the model
x = torch.randn(2, 3)
