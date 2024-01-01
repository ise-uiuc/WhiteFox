
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3, 6)
    def forward(self, x):
        z = torch.cat((x, self.linear(x)), dim=1)
        w = torch.tanh(z)
        v = torch.relu(w)
        return v
# Inputs to the model
x = torch.randn(5, 3, 4)
