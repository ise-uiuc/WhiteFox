
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lin1 = torch.nn.Linear(20, 64)
        self.lin2 = torch.nn.Linear(64, 128)
        self.lin3 = torch.nn.Linear(128, 64)
        self.lin4 = torch.nn.Linear(64, 10)
 
    def forward(self, x1, x2):
        v1 = torch.matmul(x1, x2.transpose(0, 1))
        v2 = v1 / math.sqrt(??)
        v3 = torch.nn.functional.softmax(v2, dim=-1)
        v4 = torch.nn.functional.dropout(v3, p=0.2)
        v5 = torch.matmul(v4, x2)
        v6 = self.lin1(v5)
        v7 = self.lin2(v6)
        v8 = self.lin3(v7)
        v9 = self.lin4(v8)
        return v9

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(768, 768)
x2 = torch.randn(1024, 768)
