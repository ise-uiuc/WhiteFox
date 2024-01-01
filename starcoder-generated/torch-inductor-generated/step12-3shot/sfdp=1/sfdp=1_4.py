
class Model(torch.nn.Module):
    def __init__(self, query_size, key_size):
        super().__init__()
        self.query = torch.randn(64, query_size)
        self.key = torch.randn(32, key_size)
 
    def forward(self, x1):
        qk = torch.matmul(self.query, self.key.transpose(-2, -1))
        scaled_qk = qk.div(query_size)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=dropout_p)
        output = dropout_qk.matmul(value)
        return output

# Initializing the model
m = Model(query_size=64, key_size=32)

# Inputs to the model
x1 = torch.randn(1, 64)
