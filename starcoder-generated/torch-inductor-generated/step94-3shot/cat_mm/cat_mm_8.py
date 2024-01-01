
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, input):
        return torch.cat(6 * [input], 1)
# Inputs to the model
input = torch.randn(2, 5)
