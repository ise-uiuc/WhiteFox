
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Linear(2, 1)
        self.activation = torch.sigmoid 
    def forward(self, x):
        x = self.layers(x)
        x = self.activation(x)
        return x
# Inputs to the model
x = torch.randn(2, 2)
