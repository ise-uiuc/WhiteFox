
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.fc(
            in_features=1024,
            out_features=5,
            bias=True,
            dtype=None,
            device=None,
            requires_grad=True,
        )
 
    def forward(self, x1):
        v1 = self.fc(x1)
        v2 = torch.sigmoid(v1)
        return v2

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(8, 1024)
