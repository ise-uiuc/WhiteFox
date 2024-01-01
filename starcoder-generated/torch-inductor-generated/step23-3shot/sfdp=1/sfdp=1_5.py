
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dropout_p= 0.5
        self.dropout = torch.nn.Dropout(self.dropout_p)
 
    def forward(self, q, k, v):
        qk = torch.matmul(q, k.transpose(-2, -1))
        qk = qk / math.sqrt(k.shape[-1])
        softmax_qk = qk.softmax(dim=-1)
        dropout_qk = self.dropout(softmax_qk)
        output = dropout_qk.matmul(v)
        return output

# Initializing the model
dropout_p = 0.2
m = Model(dropout_p)

# Inputs to the model
q = torch.randn(1, 3, 64, 64)
k = torch.randn(2, 3, 64, 64)
v = torch.randn(2, 3, 64, 64)
