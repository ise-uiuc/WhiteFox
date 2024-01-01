
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.qk = torch.nn.Linear(512, 512)
 
    def forward(self, x1, x2):
        v1 = self.qk(x1)
        v2 = torch.matmul(v1, x2.transpose(-2, -1))
        v3 = v2.div(math.sqrt(v1.size(-1)))
        v4 = v3.softmax(dim=-1)
        v5 = torch.nn.functional.dropout(v4, p=0.5)
        v6 = v5.matmul(x2)
        return v6

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 512)
x2 = torch.randn(1, 4, 512)
