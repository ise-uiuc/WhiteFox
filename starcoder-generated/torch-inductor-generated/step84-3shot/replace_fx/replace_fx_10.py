
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, input):
        d = torch.nn.Dropout(1)
        return d(input)
# Inputs to the model
input = torch.randn(1, 1, 2)
