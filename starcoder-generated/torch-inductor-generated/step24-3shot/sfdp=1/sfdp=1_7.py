
class Model(torch.nn.Module):
    def __init__(self, query, key, value, inv_scale_factor, dropout_p):
        super().__init__()
        self.key_transpose = key.transpose(-2, -1)
        self.inv_scale_factor = inv_scale_factor
        self.dropout_p = dropout_p
 
    def forward(self, query):
        qk = torch.matmul(query, self.key_transpose)
        scaled_qk = qk.div(self.inv_scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = dropout(softmax_qk, self.dropout_p)
        output = torch.matmul(dropout_qk, value)
        return output

# Initializing the model
value = torch.randn(32, 32, 128)
key = torch.randn(32, 32, 128)
query = torch.randn(32, 128, 25)
inv_scale_factor = torch.randn(32, 1, 25)
dropout_p = 0.8

m = Model(query, key, value, inv_scale_factor, dropout_p)

# Inputs to the model
x1 = torch.randn(32, 25, 128)
