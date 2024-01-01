
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Linear(10, 12)
        self.stack = torch.stack
        self.cat = torch.cat
    def forward(self, x):
        x = torch.flatten(x, 1)
        x = self.layers(x)
        x_list = [x, x]
        x = torch.stack(x_list)
        x = self.cat((x, x, x, x), dim=-1)
        return x
# Inputs to the model
x = torch.randn(4, 2, 5)
