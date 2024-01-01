
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.qk = torch.nn.Linear(8, 4)
 
    def forward(self, q, k, v):
        qk = self.qk(q)
        scaled_qk = qk / -0.5
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=0.5)
        o = dropout_qk.matmul(v)
        return o

# Initializing the model
q = torch.randn(2, 8)
k = torch.randn(2, 8)
v = torch.randn(4, 8)
