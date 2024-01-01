
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dropout = torch.nn.Dropout(0.2)
        self.linear = torch.nn.Linear(10, 5)
 
    def forward(self, x1):
        v1 = self.dropout(x1)
        v2 = self.linear(v1)
        v3 = torch.sigmoid(v2)
        return v3

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(32, 10)
