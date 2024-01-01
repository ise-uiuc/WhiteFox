
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Sequential(
            torch.nn.Dropout(p=0.),
            torch.nn.Linear(28*28, 1024),
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(p=0.),
            torch.nn.Linear(1024, 10)
        )
 
    def forward(self, x1, other):
        v1 = torch.flatten(x1, end_dim=1)
        v2 = self.linear(v1)
        v3 = v2 + other
        return v3

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 1, 28, 28)
other = torch.randn(1, 1, 10)
