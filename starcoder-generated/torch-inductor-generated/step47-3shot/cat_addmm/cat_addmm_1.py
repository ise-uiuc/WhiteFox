
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Linear(2, 2)
        self.activation = nn.Tanh()
    def forward(self, x):
        x = self.layers(x)
        x = self.activation(x)
        return x
# Inputs to the model
x = torch.randn(2, 2, requires_grad=True)
y = torch.randn(2, 4, requires_grad=True)
z = torch.randn(2, 5, requires_grad=True)
