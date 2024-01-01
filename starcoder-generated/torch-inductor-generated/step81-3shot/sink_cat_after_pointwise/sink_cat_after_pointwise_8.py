
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.a = torch.nn.ReLU()
        self.b = torch.nn.ReLU()
    def forward(self, x):
        y = x.view(x.size(0), -1)
        x = torch.cat((y, y), dim=1)
        x = torch.relu(x)
        x = self.a(x)
        x = self.b(x)
        return x
# Inputs to the model
x = torch.randn(2, 2, 2)
