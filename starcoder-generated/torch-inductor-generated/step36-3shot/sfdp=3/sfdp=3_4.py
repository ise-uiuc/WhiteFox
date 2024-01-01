
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.in_dim = 12
        self.out_dim = 6
        self.dropout_p = 0.2
        self.scale_factor = 1 / math.sqrt(self.in_dim)
        
    def forward(self, query, key, value):
        qk = torch.matmul(query, key.T)
        scaled_qk = self.scale_factor * qk
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=self.dropout_p)
        output = dropout_qk.matmul(value)
        return output

# Initializing the model
m = Model()

# Inputs to the model
query = torch.randn(1, 1, 12)
key = torch.randn(1, 100, 12)
value = torch.randn(1, 100, 6)
