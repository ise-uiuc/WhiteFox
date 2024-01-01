
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, input):
        return torch.exp2(input)
# Inputs to the model
input = torch.randn(3, 3)
