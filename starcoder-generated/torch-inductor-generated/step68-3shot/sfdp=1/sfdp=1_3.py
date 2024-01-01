
class Model(torch.nn.Module):
    def __init__(self, q, k, v, dropout=0.0):
        super().__init__()
        self.q = q
        self.k = k
        self.v = v
        self.dropout = dropout
 
    def forward(self):
        qk = torch.matmul(self.q, self.k.transpose(-2, -1))
        inv_scale_factor = max(float(self.k.shape[-1]), 1.0)
        scaled_qk = qk.div(inv_scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=self.dropout)
        output = dropout_qk.matmul(self.v)
        return output

# Intializing the query, key and value tensors as well as the dropout parameter
q = torch.randn(1, 5, 64, 64)
k = torch.randn(1, 5, 64, 64)
v = torch.randn(1, 5, 64, 64)
dropout = 0.1

# Initializing the model
m = Model(q, k, v, dropout)

# Applying the model to the query tensor
