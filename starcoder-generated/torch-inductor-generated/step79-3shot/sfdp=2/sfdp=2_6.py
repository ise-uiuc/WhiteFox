
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, x_query, x_key, x_value, mask, dropout_p, inv_scale_factor):
        qk = x_query.matmul(x_key.transpose(-2, -1))
        scaled_qk = qk / inv_scale_factor
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=dropout_p)
        output = dropout_qk.matmul(x_value)
        return output

# Initializing the model
m = Model()

# Inputs to the model
x_query = torch.randn(1, 1, 5, 7)
x_key = torch.randn(1, 1, 7, 7)
x_value = torch.randn(1, 1, 7, 5)
mask = torch.randint(0, 1, [1, 1, 1, 5])
dropout_p = 0.5
inv_scale_factor = 10.0
