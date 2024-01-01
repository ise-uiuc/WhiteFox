
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1):
        x1 = torch.cat((x1, x1), dim=-1)
        x1 = x1.view(2, 3, 4)
        return x1.relu()
# Inputs to the model
x1 = torch.randn(1, 2, 2)
