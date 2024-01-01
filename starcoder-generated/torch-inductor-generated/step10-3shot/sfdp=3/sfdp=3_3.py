
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(64, 4)
        self.dropout = torch.nn.Dropout()
        self.linear2 = torch.nn.Linear(4, 64)
 
    def forward(self, x1):
        v1 = self.linear1(x1)
        v2 = self.dropout(v1)
        v3 = self.linear2(v2)
        return v3

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(2, 64)
