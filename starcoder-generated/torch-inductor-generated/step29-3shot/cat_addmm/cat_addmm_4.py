
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Linear(2, 2)
    def forward(self, x):
        x = self.layers(x)
        dim2 = 2
        dim1 = 2
        x = x.flatten(start_dim=1)
        x = torch.stack([x], dim=0)
        x = x.reshape(2*x.size(dim1)+x.size(dim2), 1)
        return x
# Inputs to the model
x = torch.randn(2, 2)
