
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Linear(2, 1)
    def forward(self, x):
        x = self.layers(x)
        x[:, 0].flatten(0)
        return x
# Inputs to the model
x = torch.randn(2, 2)
y = torch.Tensor([[0.6003]])

