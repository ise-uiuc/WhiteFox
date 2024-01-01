
class Model2(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, query, key, value, scale_factor, dropout_p):
        qk = torch.matmul(query, key.transpose(-2, -1))
        scaled_qk = qk.mul(scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=dropout_p)
        output = dropout_qk.matmul(value)
        return output

# Initializing the model2
m = Model2()

# Inputs to the model2
query = torch.randn(1, 10, 64)
key = torch.randn(1, 10, 64)
value = torch.randn(1, 10, 64)
scale_factor = 10.0 ** np.random.uniform(-1, 1)
dropout_p = np.random.uniform(0, 1)
