
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, q1, k2, v3):
        qk = torch.matmul(q1, k2.transpose(-2, -1))
        scale_factor = torch.rsqrt(torch.tensor(self.head_dim).float())
        v4 = qk.mul(scale_factor)
        softmax_qk = v4.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=dropout_p)
        v6 = dropout_qk.matmul(v3)
        return v6

# Initializing the model
m = Model()

# Input tensors to the model
q1 = torch.randn(head_dim, input_len, key_len)
k2 = torch.randn(head_dim, key_len, query_len)
v3 = torch.randn(head_dim, value_len, query_len)
