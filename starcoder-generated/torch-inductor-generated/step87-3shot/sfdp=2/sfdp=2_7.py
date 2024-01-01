
class Model(torch.nn.Module):
    def __init__(self, d, n0, n1, n2, dropout_p=0.0):
        super().__init__()
        self.dropout_p = dropout_p
        self.layers = torch.nn.ModuleList([
            torch.nn.Linear(d, n0),
            torch.nn.Linear(n0, n1),
            torch.nn.Linear(n1, n2),
        ])
 
    def forward(self, x1, x2, x3):
        v1 = self.layers[0](x1)
        v1 = torch.relu(v1)
        v2 = self.layers[1](v1)
        v2 = torch.relu(v2)
        v3 = torch.matmul(v2, x2)
        v3 = torch.relu(v3)
        v4 = self.layers[2](v3)
        v4 = torch.relu(v4)
        return v4

# Initializing the model
m = Model(d=8, n0=4, n1=8, n2=8)
 
# Inputs to the model
x1 = torch.randn(16, 8)
x2 = torch.randn(4, 8)
x3 = torch.randn(8, 8)
