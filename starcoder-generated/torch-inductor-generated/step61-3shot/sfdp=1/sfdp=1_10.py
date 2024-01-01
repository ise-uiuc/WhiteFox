
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, query, key, value):
        qk = query @ key.transpose(-2, -1)
        inv_scale = 1.0/(len(key) ** 0.5)
        scaled_qk = qk * inv_scale
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=0.3)
        output = dropout_qk @ value
        return output

# Initializing the model
m = Model()

# Inputs to the model
query = torch.randn(1, 16, 64)
key   = torch.randn(1, 32, 64)
value = torch.randn(1, 32, 64)
