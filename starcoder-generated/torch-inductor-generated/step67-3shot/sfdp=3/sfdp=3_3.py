
class Model(torch.nn.Module):
    def __init__(self, dropout=0):
        super().__init__()
        self.dropout = dropout
 
    def forward(self, query, key, value, scale_factor=1.0):
        qk = torch.matmul(query, key.transpose(-2, -1))
        scaled_qk = qk.mul(scale_factor)
        self.softmax_qk = scaled_qk.softmax(dim=-1)
        if self.dropout > 0.0:
            dropout_qk = torch.nn.functional.dropout(self.dropout_qk, p=dropout_p)
 
        output = dropout_qk.matmul(value)
        return output

# Initializing the model
m = Model(dropout=0.1)

# Inputs to the model
query = torch.randn(1, 4, 20)
key = torch.randn(1, 5, 20)
value = torch.randn(1, 5, 20)
