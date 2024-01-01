
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, query, key, value, inv_scale_factor, dropout_p):
        kq = torch.matmul(query, key.transpose(-2, -1))
        drop_kq = torch.nn.functional.dropout(kq, p=dropout_p)
        scaled_drop_kq = drop_kq.div(inv_scale_factor)
        output = scaled_drop_kq.matmul(value)
        return output

# Initializing the model
m = Model()

# Inputs to the model
query = torch.randn(1, 8, 3136)
key = torch.randn(1, 16, 3136)
value = torch.randn(1, 16, 3136)
inv_scale_factor = torch.tensor(1.0, dtype=torch.float)
dropout_p = 0.2
