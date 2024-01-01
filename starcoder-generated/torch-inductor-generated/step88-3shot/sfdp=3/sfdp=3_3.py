
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dropout_p = 0
 
    def forward(self, query, key, value, scale_factor, dropout_p):
        qk = torch.matmul(query, key.transpose(-2, -1))
        scaled_qk = qk.mul(scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=dropout_p)
        output = dropout_qk.matmul(value)
        return output

# Initializing the model
m = Model()

# Inputs to the model
query  = torch.randn(1, 1, 128, 64)
key    = torch.randn(1, 1, 128, 64)
value  = torch.randn(1, 1, 128, 64)
scale_factor = 5
dropout_p = 0.1
