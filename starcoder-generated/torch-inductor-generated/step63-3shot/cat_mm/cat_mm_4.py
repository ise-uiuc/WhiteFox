
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, inputs):
        v = torch.mm(inputs, inputs)
        return torch.cat([v, v, v, v], 0)
# Inputs to the model
inputs = torch.randn(2, 2)
