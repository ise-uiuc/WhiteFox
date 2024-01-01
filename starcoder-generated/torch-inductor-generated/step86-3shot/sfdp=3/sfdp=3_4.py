

class Model(torch.nn.Module):
    def __init__(self)
        super().__init__()
        self.dropout = torch.nn.Dropout(p)
          
    def forward(self, query, key, value, scaling_factor=1):
        qk = torch.matmul(query, key.transpose(-2, -1))
        scaled_qk = qk.mul(scaling_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = self.dropout(softmax_qk)
        output = dropout_qk.matmul(value)
        return output

# Initializing the model
m = Model()

# Inputs to the model
query = torch.randn(1, 128, 64, 64)
key =  torch.randn(1, 128, 64, 64)
value = torch.randn(1, 128, 64, 64)
scaling_factor = 1.0
p = 0.1
