
class Model(nn.Module):
    # Here is an example of using argument value which is equal to False in a function
    def __init__(self):
        super().__init__()
        self.layers = nn.Linear(4, 5, False)
    def forward(self, x):
        x = self.layers(x)
        x = torch.cat([x], dim=1)
        x = torch.flatten(x, start_dim=1)
        return x
# Inputs to the model
x = torch.randn(1, 4)
