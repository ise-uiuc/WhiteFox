
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, query, key, value, scale_factor, dropout_p):
        qk = torch.matmul(query, key.transpose(-2, -1))
        scaled_qk = qk.mul(scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout = torch.nn.functional.dropout(softmax_qk, p=dropout_p)
        output = dropout.matmul(value)
        return output

# Initializing the model
m = Model()

# Inputs to the model
query = torch.randn(1, 32, 50)
key = torch.randn(1, 2, 50)
value = torch.randn(1, 2, 100)
scale_factor = torch.full((1, 2, 50), 0.1)
dropout_p = 0.1
