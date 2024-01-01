
class Model(torch.nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
 
    def forward(self, query, key, value, scale_factor, dropout_p):
        qk = torch.matmul(query, key.transpose(-2, -1))
        scaled_qk = qk.mul(scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=dropout_p)
        output = dropout_qk.matmul(value)
        return output

# Initializing the model
dim = 4
m = Model(dim)

# Inputs to the model
query = torch.randn(4, 32, dim)
key = torch.randn(8, 64, dim)
value = torch.randn(8, 64, dim)
scale_factor = torch.tensor([0.0625])
dropout_p = torch.tensor([0.1])
