
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.scale_factor = nn.Parameter(torch.ones(1) * 1.0 /math.sqrt(query.shape[-1]))
 
    def forward(self, query, key, value, dropout_p=0.0):
        qk = torch.matmul(query, key.transpose(-2, -1))
        scaled_qk = qk.mul(self.scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=dropout_p)
        output = dropout_qk.matmul(value)
        return output

# Initializing the model
m = Model()

# Inputs to the model
query = torch.randn(1, 2, 3, 4)
key = torch.randn(1, 2, 5, 6)
value = torch.randn(1, 2, 5, 6)
