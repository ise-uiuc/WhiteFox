
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Linear(5, 5)
    def forward(self, x):
        x = x.squeeze(1)
        x = self.layers(x)
        x = x.unsqueeze(1)
        return x
# Inputs to the model
x = torch.randn(1, 2, 5)
