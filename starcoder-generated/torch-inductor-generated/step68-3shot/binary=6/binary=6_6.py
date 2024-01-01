
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(300, 400)
        self.dropout = torch.nn.Dropout()
 
    def forward(self, x1, x2):
        v1 = self.linear(x1)
        v2 = v1 - x2
        v3 = self.dropout(v2)
        return v3

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 84, 300)
x2 = x1 * 0.35
