
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.query = torch.nn.Linear(16, 4)
        self.key = torch.nn.Linear(16, 4)
 
    def forward(self, x1):
        v1 = self.query(x1)
        v2 = self.key(x1)
        v3 = torch.matmul(v1, v2.transpose(-2, -1))
        v4 = v3.div(0.09216988578950622)
        v5 = v4.softmax(dim=-1)
        v6 = torch.nn.functional.dropout(v5, p=0.15000000596046448)
        v7 = v6.matmul(v1)
        return v7

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(3, 16)
