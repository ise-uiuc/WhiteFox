
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, __input__):
        scale_factor = 1.0 / math.sqrt(query.size(-1))
        qk = torch.matmul(query, key.transpose(-2, -1))
        scaled_qk = qk.mul(scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=dropout_p)
        output = dropout_qk.matmul(value)
        return output

# Initializing the model
m = Model()

# Inputs to the model
query = torch.randn(1, 2, 3)
key = torch.randn(1, 3, 5)
value = torch.randn(1, 3, 5)
