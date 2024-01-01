
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        a = x.tanh()
        return torch.cat((a, a + 1), dim=1)
# Inputs to the model
x = torch.randn(2, 2, 2)
