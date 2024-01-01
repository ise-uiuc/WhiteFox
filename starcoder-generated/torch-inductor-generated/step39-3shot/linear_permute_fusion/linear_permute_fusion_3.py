
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 2)
    def forward(self, x1):
        conv1 = torch.nn.ConvTranspose2d(2, 2, 2)
        v1 = conv1(x1)
        v2 = torch.nn.functional.linear(x1, self.linear.weight, self.linear.bias)
        return v2
# Inputs to the model
x1 = torch.randn(1, 2, 2, 3)
