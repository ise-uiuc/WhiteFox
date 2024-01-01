
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.scale_factor = np.power(d_k, 0.5)
 
    def forward(self, query, key, value, dropout_p):
        qk = torch.matmul(query, key.transpose(-2, -1))
        scaled_qk = qk.mul(self.scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=dropout_p)
        output = dropout_qk.matmul(value)
        return output

# Initializing the model
dropout_p = 0.1
d_k = 1024
m = Model()

# Inputs to the model
query = torch.randn(1024, 1, d_k)
key = torch.randn(1024, 40, d_k)
value = torch.randn(1024, 40, d_k)
