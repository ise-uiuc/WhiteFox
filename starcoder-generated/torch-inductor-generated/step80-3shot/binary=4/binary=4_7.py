
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(in_features=512, out_features=256, bias=True)

    def forward(self, x1, x2):
        v1 = self.fc(x1)
        v2 = v1 + x2
        return v2

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 512)
x2 = torch.randn(1, 256)
