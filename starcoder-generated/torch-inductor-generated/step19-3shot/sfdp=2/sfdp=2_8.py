
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
       
    def forward(self, query, key, value, inv_scale_factor, dropout_p):
        qk = torch.matmul(query, key.transpose(-2, -1))
        scaled_qk = qk.div(inv_scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=dropout_p)
        output = dropout_qk.matmul(value)
        return output

# Initializing the model
m = Model()

# Inputs to the model
dropout_p = 0.5
query = torch.randn(2, 4, 64)
key = torch.randn(2, 4, 128)
value = torch.randn(2, 128, 64)
inv_scale_factor = torch.tensor(1.0)
