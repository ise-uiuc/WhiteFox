
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def compute_similarity(self, query, key):
        return torch.matmul(query, key.transpose(-2, -1))
 
    def forward(self, query, key, value, scale_factor, dropout_p):
        qk = self.compute_similarity(query, key)
        inv_scale_factor = 1.0 / scale_factor
        scaled_qk = qk.div(inv_scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=dropout_p)
        return dropout_qk.matmul(value)

# Initializing the model
m = Model()

# Inputs to the model
query = torch.randn(1, 8, 8, 64)
key = torch.randn(1, 8, 8, 64)
value = torch.randn(1, 8, 8, 64)
scale_factor = 64.0
dropout_p = 0.0
