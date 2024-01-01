
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.softmax = torch.nn.Softmax(dim=-1)
        self.dropout = torch.nn.Dropout(dropout_p)
 
    def forward(self, q, k, v):
        qk = torch.matmul(q, k.transpose(-2, -1))
        scaled_qk = qk * scale_factor
        softmax_qk = self.softmax(scaled_qk)
        drop_qk = self.dropout(softmax_qk)
        output = drop_qk.matmul(v)
        return output

# Initializing the model
m = Model()

# Initialize inputs
query = torch.randn(1, 32, 256)
key = torch.randn(1, 8, 256)
value = torch.randn(1, 8, 256)

# Running the model
__output__=m(query, key, value)

