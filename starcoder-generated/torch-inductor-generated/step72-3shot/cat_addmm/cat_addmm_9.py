
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Linear(1, 1)
    def forward(self, x):
        x = self.layers(x)
        x = x.unsqueeze(0)
        x = x.unsqueeze(0)
        x = x.unsqueeze(0)
        return x
# Inputs to the model
x = torch.randn(1, 1)
