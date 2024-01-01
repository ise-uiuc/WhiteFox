
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Linear(3, 4)
        self.layer2 = nn.ReLU()
        self.layer3 = nn.Linear(4, 5)
    def forward(self, x):
        x = self.layers(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = x.flatten(start_dim=1)
        return x
# Inputs to the model
x = torch.randn(3, 3)
