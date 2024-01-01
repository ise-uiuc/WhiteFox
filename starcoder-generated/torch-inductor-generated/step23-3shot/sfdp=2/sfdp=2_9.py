
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = torch.nn.Linear(16, 32, bias=False)
        self.scale = torch.nn.Parameter(torch.tensor([2000.0]))
 
    def forward(self, x1):
        v1 = self.embed(x1)
        v2 = v1.transpose(-2, -1)
        v3 = torch.matmul(v1, v2)
        v4 = self.scale.expand_as(v3)
        v5 = v3.div(v4)
        v6 = torch.nn.functional.softmax(v5, -1)
        v7 = torch.nn.functional.dropout(v6, p=0.2, training=True)
        v8 = torch.matmul(v7, v1)
        return v8

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(2, 16)
