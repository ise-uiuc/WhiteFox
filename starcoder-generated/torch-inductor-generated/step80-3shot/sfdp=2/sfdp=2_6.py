
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(12, 16)
        self.dropout = torch.nn.Dropout(p=0.8)
 
    def forward(self, q, k, v):
        qkt = torch.matmul(q, k.transpose(-2, -1))
        qkt = qkt / (1000 ** 0.5)
        qkt = torch.nn.functional.softmax(qkt, dim=-1)
        qkt = torch.nn.functional.dropout(qkt, p=0.8)
        output = torch.matmul(qkt, v)
        return output

# Initializing the model
m = Model()

# Inputs to the model
q = torch.randn(64, 12)
k = torch.randn(64, 8, 12)
v = torch.randn(64, 8, 12)
