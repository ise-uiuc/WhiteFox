
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, input):
        x1.add_(input)
        return x1

# Inputs to the model
x1 = torch.randn(10, 10)
x = torch.randn(10, 10)
