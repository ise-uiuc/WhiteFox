
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(64 * 64 * 3, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 10)
        )
 
    def forward(self, x1, x2):
        v1 = self.layers[0](x2)
        v2 = v1 + x1
        for layer in self.layers[1:]:
            v1 = layer(v2)
            v2 = v1 + v2
        return v2

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 64 * 64 * 3)
x2 = torch.randn(1, 128)
