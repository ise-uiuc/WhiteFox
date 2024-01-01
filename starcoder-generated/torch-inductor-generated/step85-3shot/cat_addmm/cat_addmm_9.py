
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Linear(2, 7)
    def forward(self, x):
        x = self.layers(x)
        x = x.view(4, 2)
        x = x.flatten(start_dim=0, end_dim=1)
        return x
# Inputs to the model
x = torch.randn(2, 2)
