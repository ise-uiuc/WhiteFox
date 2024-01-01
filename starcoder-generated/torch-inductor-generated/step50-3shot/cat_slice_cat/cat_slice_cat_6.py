
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = torch.nn.Sequential(
            torch.nn.Conv2d(3, 8, 1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(8, 16, 1)
        )
 
    def forward(self, x1):
        v1 = self.model(x1)
        v2 = torch.cat(list(torch.unbind(v1, dim=1)), dim=1)
        v3 = v2[:, 0:9223372036854775807]
        v4 = v3[:, 0:x1.size(2)]
        v5 = torch.cat((v1, v4), dim=1)
        return v5

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(8, 3, 64, 64)
