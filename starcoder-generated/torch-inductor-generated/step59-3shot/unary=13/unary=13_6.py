
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(16, 3)
        self.dropout = torch.nn.Dropout(p=0.3)
 
    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = torch.sigmoid(v1)
        v3 = v1 * v2
        v4 = self.dropout(v1)
        v5 = v4 * v2
        return v5, v2
# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
o1, o2 = m(x1)

# Please change this line to output the value of o1 and o2
