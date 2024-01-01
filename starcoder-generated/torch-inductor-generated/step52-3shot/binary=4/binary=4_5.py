
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.linear = torch.nn.modules.linear.Linear(20, 40)

    def forward(self, x):
        return self.linear(x) + x.mean()

# Initializing the model
s, m = 20, 40
x = torch.randn(1, s, m, dtype=torch.float32)

model = Model()
y = model(x)

