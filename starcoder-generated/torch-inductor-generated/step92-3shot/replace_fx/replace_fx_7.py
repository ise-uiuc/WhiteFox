
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, input):
        return input.transpose(0, 1).flatten(1)
# Inputs to the model
input = torch.randn(1, 2, 2)
