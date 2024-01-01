
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Linear(10, 5)
    def forward(self, x):
        input = torch.cat((x, x, x, x, x, x, x, x, x, x), dim=1)
        x = self.layers(input)
        return x
# Inputs to the model
x = torch.randn(2, 10)
