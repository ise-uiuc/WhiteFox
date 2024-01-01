
class Model(torch.nn.Module):
    def __init__(self, key_size):
        super().__init__()
        self.key_size = key_size
    
    def forward(self, query, value, scale_factor, dropout_p):
        dropout_qk = torch.matmul(query, value.transpose(-2, -1) / scale_factor).softmax(dim=-1).div(scale_factor).dropout(p=dropout_p).matmul(value)
        return dropout_qk

# Initializing the model
m = Model(key_size)

# Inputs to the model
query = torch.randn(1, 4, key_size)
value = torch.randn(1, 16, key_size)
scale_factor = 320
dropout_p = 0.5
