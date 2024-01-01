
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.weight = torch.randn(2, 3, 4)
        self.bias = torch.randn(2, 2)

    def forward(self, x):
        y1 = torch.cat((self.weight, x), dim=1)
        y2 = torch.relu(y1)
        return y2
# Inputs to the model
x = torch.randn(2, 3, 4)
