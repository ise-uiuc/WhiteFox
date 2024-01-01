
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.head_dim = 64
 
    def forward(self, q, k, v):
        query = q.reshape(-1, 1, 1, self.head_dim)
        key   = k.reshape(-1, 1, 1, self.head_dim)
        value = v.reshape(-1, 1, 1, self.head_dim)
        softmax_qk = torch.matmul(query, key.transpose(-2, -1))
        scale_factor = softmax_qk.size(-1) ** -0.5
        softmax_qk = softmax_qk * scale_factor
        softmax_qk = torch.nn.functional.softmax(softmax_qk, dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=0.1)
        output = dropout_qk.matmul(value)
        return output.reshape(-1, 1024)

# Initializing the model
m = Model()

# Inputs to the model
q = torch.randn(1, 1, 1, 1024)
k = torch.randn(1, 1, 1, 1024)
v = torch.randn(1, 1, 1, 1024)
