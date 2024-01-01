
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, input, shape):
        x = input.view(*shape)
        x = x.sum()
        return x
# Inputs to the model
input = torch.randn(20, 100)
shape = (2, 5, 10)
