
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self,  value, key, query, inv_scale_factor, dropout_p):
        qk = torch.matmul(query, key.transpose(-2, -1))
        scaled_qk = qk.div(inv_scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=dropout_p)
        return dropout_qk.matmul(value)

# Initializing the model
m = Model()

# Inputs to the model
value = torch.randn(8, 1, 25, 64)
key   = torch.randn(8, 1, 25, 64)
query = torch.randn(8, 1, 25, 64)
inv_scale_factor = torch.randn(8)
dropout_p=0.07542524
