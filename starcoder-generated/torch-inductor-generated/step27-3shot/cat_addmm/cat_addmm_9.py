
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Linear(2, 4)
    def forward(self, x):
        pass
# Inputs to the model
x = torch.randn(3, 2)
