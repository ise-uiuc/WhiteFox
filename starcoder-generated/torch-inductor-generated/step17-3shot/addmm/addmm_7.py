
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, *inputs):
        return torch.cat(inputs, dim=1)
# Inputs to the model
x1 = torch.randn(3, 10)
x2 = torch.randn(10, 3)
inp = torch.randn(5, 10)
