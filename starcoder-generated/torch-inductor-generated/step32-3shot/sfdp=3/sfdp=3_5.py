
class Model(torch.nn.Module):
    def __init__(self, n):
        super(Model, self).__init__()
        self.num_heads = int(n / 6)
 
    def forward(self, query, key, value, dropout_p, scale_factor):
        qk = torch.matmul(query, key.transpose(-2, -1))
        scaled_qk = qk.mul(scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=dropout_p)
        output = dropout_qk.matmul(value)
        return output

# Initializing the model
n = 12
d_model = 6
dropout_p = 0.1
scale_factor = math.sqrt(d_model)

m = Model(n)

# Input of the model
query = torch.randn(2, 1, d_model, n)
key = torch.randn(2, 1, d_model, n)
value = torch.randn(2, 1, d_model, n)

# Outputs of the model
