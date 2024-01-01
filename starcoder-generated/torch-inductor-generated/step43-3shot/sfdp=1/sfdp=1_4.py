
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dropout_p = 0.
    def forward(self, x2):
        v2 = x2[0]
        v3 = x2[1]
        v4 = (v2[0] - v2[2]).abs()
        v5 = v4 * v3
        v6 = v2[1]
        v7 = v2[1] - v2[2]
        
        v8 = torch.nn.functional.relu(v5)
        v9 = v8 + v6
        v11 = torch.nn.functional.relu(v7)
        v12 = v9 / v11
        return v12


# Initializing the model
m = Model()

# Inputs to the model
x2 = [torch.randn(3, 3, 64, 64), torch.randn(3, 3, 64, 64), torch.randn(3, 3, 64, 64)]
