
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers1 = nn.Linear(3, 3)
        self.layers2 = nn.Linear(3, 3)
    def forward(self, x):
        x = self.layers1(x)
        x = self.layers2(x)
        x = (torch.flatten(x, start_dim=0), )
        return x
# Inputs to the model
x = torch.randn(2, 2)
