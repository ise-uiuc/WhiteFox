
d_k = 2048
class Model(torch.nn.Module):
    def __init__(self, dropout_p):
        super().__init__()
        self.dropout_p = dropout_p
 
    def forward(self, query, key, value, scale_factor):
        qk = torch.matmul(query, key.transpose(-2, -1))
        inv_scale_factor = 1 / scale_factor.unsqueeze(1)
        scaled_qk = qk * inv_scale_factor.unsqueeze(1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=dropout_p)
        return torch.matmul(dropout_qk, value)

# Initializing the model
dropout_p = 0.1
m = Model(dropout_p)

# Inputs to the model
query = torch.randn(768, 1, d_k)
key = torch.randn(768, 100, d_k)
value = torch.randn(768, 100, d_k)
scale_factor = torch.tensor([10.0])
