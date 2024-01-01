
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        y = torch.randn(6)
        x = torch.cat((y.unsqueeze(1).expand(-1, 3, -1), x), dim=1).view(-1)
        return x
# Inputs to the model
x = torch.randn(3, 4)
