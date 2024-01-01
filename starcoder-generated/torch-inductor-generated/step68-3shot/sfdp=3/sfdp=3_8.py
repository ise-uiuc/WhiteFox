
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.q = torch.nn.Linear(2, 5)
        self.k = torch.nn.Linear(3, 6)
        self.v = torch.nn.Linear(3, 8)
    
    def forward(self, q, k, v):
        qk = torch.matmul(q, k.transpose(-2, -1))
        scale_factor = torch.tensor(1 / math.sqrt(q.size(-1)))
        scaled_qk = qk * scale_factor
        softmax_qk = torch.nn.functional.softmax(scaled_qk, -1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=0.4)
        out = torch.matmul(dropout_qk, v)
        return out

# Initializing the model
model = Model()

# Inputs to the model
q = torch.randn(2, 2)
k = torch.randn(1, 3)
v = torch.randn(30, 3)
