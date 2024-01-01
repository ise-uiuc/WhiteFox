
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, input):
        x = torch.nn.ReLU(inplace=True)(input)
        x = x - 3
        x = torch.nn.functional.relu6(x)
        x = x / 6
        return x
# Inputs to the model
x = torch.randn(1, 3, 64, 64)
