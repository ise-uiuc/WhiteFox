
class Model(torch.nn.Module):
    def __init__(self, dropout_p, hidden_size, scale_factor):
        super().__init__()
        self.dropout_p = dropout_p
        self.hidden_size = hidden_size
        self.scale_factor = scale_factor
    
    def forward(self, query, key, value):
        bz = query.size(0)
        qk = torch.matmul(query, key.transpose(-2, -1))
        scaled_qk = qk.mul(scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=dropout_p)
        output = dropout_qk.matmul(value)
        return output

# Initializing the model
m = Model(0.1, 128, 1/math.sqrt(128))

# Inputs to the model
query = torch.randn(30, 45, 128)
key = torch.randn(30, 45, 128)
value = torch.randn(30, 45, 128)
