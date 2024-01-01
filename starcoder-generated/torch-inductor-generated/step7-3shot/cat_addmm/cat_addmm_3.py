
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(100, 100)
        self.relu = torch.nn.ReLU(inplace=False)
        self.dropout = torch.nn.Dropout(p=0.1)
 
    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = self.relu(v1)
        v3 = self.dropout(v2)
        v4 = torch.cat([v3], dim=1)
        return v4

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 100, 10)
