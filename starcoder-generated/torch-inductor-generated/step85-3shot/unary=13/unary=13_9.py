
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 2, 3, stride=1, padding=1)
        self.linear = torch.nn.Linear(18, 5)
 
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = v1.reshape(-1)
        v3 = self.linear(v2)
        v4 = torch.sigmoid(v3)
        v5 = v3 * v4
        return v5

# Initializing the model
m = Model()

# Initializing the optimizer
optimizer = torch.optim.SGD(
    m.parameters(),
    lr=0.01,
    momentum=0.9,
    weight_decay=0.0001
)

# Inputs to the model
x1 = torch.randn(10, 3, 64, 64)
