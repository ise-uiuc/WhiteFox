
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, q, k, v, scale_factor, dropout_p):
        qk = torch.matmul(q, k.transpose(-2, -1))
        scaled_qk = qk.mul(scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=dropout_p)
        output = dropout_qk.matmul(v)

# Initializing the model
q = torch.randn(1, 3, 8, 8)
k = torch.randn(1, 3, 8, 8)
v = torch.randn(1, 3, 8, 8)
scale_factor = 1 / math.sqrt(8)
dropout_p = 0.5
