
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        return torch.cat((x, x, x), dim=0).tanh() if torch.sum(x).item()!= 0 else torch.cat((x, x, x), dim=0).relu()
# Inputs to the model
x = torch.randn(2, 2, 2)
