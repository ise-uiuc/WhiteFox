
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.linear_layer = torch.nn.Linear(2, 8)
 
    def forward(self, x1):
        v1 = self.linear_layer(x1)
        v2 = torch.sigmoid(v1)
        v3 = v1 * v2
        return v3

# Initializing the model
m = Model()

# Input to the model
x1 = torch.randn(1, 2, 3, 4)

# Output of the model
