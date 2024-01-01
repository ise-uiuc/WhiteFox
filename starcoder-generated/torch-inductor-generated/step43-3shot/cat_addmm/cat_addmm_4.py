
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers_1 = nn.Linear(2, 4)
        self.layers_2 = nn.Linear(2, 4)
        self.layers = nn.Linear(4, 4)
    def forward(self, x):
        print("x:", x.shape)
        x = self.layers_1(x)
        print("After layer_1:", x.shape)
        x = self.layers_2(x)
        print("After layer_2:", x.shape)
        x = self.layers(x)
        return x
# Inputs to the model
x = torch.randn(2, 2)
