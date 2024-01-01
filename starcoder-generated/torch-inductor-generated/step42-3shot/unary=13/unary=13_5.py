
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(4, 4, bias=True)
    def forward(self, x):
        y = self.linear(x)
        y = torch.sigmoid(y)
        y = self.linear(torch.cat([x, y], 1))
        y = torch.sigmoid(y)
        y = self.linear(torch.cat([x, y], 1))
        y = torch.sigmoid(y)
        y = self.linear(torch.cat([x, y], 1))
        return y

# Initializing the model
batch_size = 4
features = 4
model = Model()

# Inputs to the model
x = torch.randn(batch_size, features)
