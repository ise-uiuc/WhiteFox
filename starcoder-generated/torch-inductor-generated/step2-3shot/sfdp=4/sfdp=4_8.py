
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.head = torch.nn.TransformerEncoderLayer(4, 4, 4, 4, 2)

    def forward(self, x1):
        v1 = self.head(x1)
        return v1

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(3, 5, 4)
