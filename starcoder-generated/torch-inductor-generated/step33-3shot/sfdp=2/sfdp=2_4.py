
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, query, key, value, inv_scale_factor):
        qk = torch.matmul(query, key.transpose(-2, -1))
        scaled_qk = qk.div(inv_scale_factor)
        dropout_qk = torch.nn.functional.dropout(scaled_qk.softmax(dim=-1), p=1-0.75)
        output = dropout_qk.matmul(value)
        return output

# Initializing the model
m = Model()

# Inputs to the model
query = x = torch.randn(1, 10, 4)
key = k = torch.randn(1, 10, 8)
value = v = torch.randn(1, 10, 8)
inv_scale_factor = z = torch.tensor(0.03125)
