
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, query, key, value, dropout_p):
        qk = torch.matmul(query, key.transpose(-2, -1))
        inv_scale_factor = (1.0 / (query.size(-2) ** 0.5))
        scaled_qk = qk.div(inv_scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=dropout_p)
        return dropout_qk.matmul(value)

dropout_p = 0.0

# Initializing the model
m = Model()

# Tensors to the model
query = torch.randn(10, 100, 64, 32)
key = torch.randn(10, 200, 32, 16)
value = torch.randn(10, 200, 32, 16)
