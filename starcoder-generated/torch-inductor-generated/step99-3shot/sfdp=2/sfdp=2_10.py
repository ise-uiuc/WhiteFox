
class Model(torch.nn.Module):
    def __init__(self, query_dim, key_dim, value_dim):
        super().__init__()
        self.scale_factor = torch.sqrt(torch.tensor(query_dim))
 
    def forward(self, q, k, v, dropout_p):
        qk = torch.matmul(q, k.transpose(-2, -1))
        scaled_qk = qk.div(self.scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=dropout_p)
        output = dropout_qk.matmul(v)
        return output

# Initializing the model
m = Model(query_dim=3, key_dim=6, value_dim=8)

# Inputs to the model
query = torch.randn(1, 1, 3)
key = torch.randn(1, 1, 6)
value = torch.randn(1, 1, 8)
dropout_p = 0.5
