
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def scaled_dot_product(self, query, key, scale_factor):
        qk = torch.matmul(query, key.transpose(-2, -1))
        scaled_qk = qk.mul(scale_factor)
        return scaled_qk
 
    def attention(self, query, key, value, scale_factor, dropout_p):
        softmax_qk = self.scaled_dot_product(query, key, scale_factor).softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=dropout_p)
        output = dropout_qk.matmul(value)
        return output
 
    def forward(self, query, key, value, scale_factor, dropout_p):
        output = self.attention(query, key, value, scale_factor, dropout_p)
        return output

# Initializing the model
m = Model()

# Inputs to the model
query = torch.randn(1, 50, 128)
key = torch.randn(1, 32, 100)
value = torch.randn(1, 32, 160)
scale_factor = torch.tensor(0.1)
dropout_p = torch.tensor(0.1)
