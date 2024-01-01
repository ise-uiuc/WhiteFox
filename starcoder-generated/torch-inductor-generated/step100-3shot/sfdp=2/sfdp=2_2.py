
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.softmax = torch.nn.Softmax(dim=1)
 
    def forward(self, q, k, v):
        qk = torch.matmul(q, k.transpose(-2, -1))
        dropout_qk = torch.nn.functional.dropout(self.softmax(qk), p=0.1, training=True)
        return torch.matmul(dropout_qk, v)

# Initializing the model
m = Model()

# Inputs to the model
q = torch.randn(1, 6, 8)
k = torch.randn(1, 6, 8)
v = torch.randn(1, 6, 8)
