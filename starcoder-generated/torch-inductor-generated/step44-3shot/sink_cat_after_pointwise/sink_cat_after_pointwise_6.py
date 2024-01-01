
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        x = x.unsqueeze(1)
        return torch.cat((x, x, x), dim=0)
# Inputs to the model
x = torch.randn(2, 2)
