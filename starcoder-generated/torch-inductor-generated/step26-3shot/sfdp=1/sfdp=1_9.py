
class Model(torch.nn.Module):
    def __init__(self, dropout):
        super().__init__()
        self.dropout = dropout
 
    def forward(self, q, k, v, scale_factor):
        k = k.transpose(-2, -1)
        qk = torch.matmul(q, k)
        inv_scale_factor = 1.0 / scale_factor
        scaled_qk = qk.div(inv_scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=self.dropout)
        output = dropout_qk.matmul(v)
        return output

# Initializing the model
dropout_p = 0.75
m = Model(dropout_p)

# Inputs to the model
q = torch.randn(2, 8, 64)
k = torch.randn(2, 4, 128)
v = torch.randn(2, 4, 128)
scale_factor = torch.randn(1)
