
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.scale_factor = torch.nn.Parameter(torch.tensor([math.sqrt(16) / math.sqrt(15)]))
 
    def forward(self, query, key, value, dropout_p):
        qk = torch.matmul(query, key.transpose(-2, -1))
        scaled_qk = qk.mul(self.scale_factor)
        return value.matmul(scaled_qk.transpose(-2, -1))

# Initializing the model
m = Model()

# Inputs to the model
query = torch.randn(2, 3, 4)
key = torch.randn(2, 4, 5)
value = torch.randn(2, 5, 6)
dropout_p = 0.1
