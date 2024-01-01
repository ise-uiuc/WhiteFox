
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        hidden_size = 4
        drop_p = 0.3
        scale_factor = math.sqrt(hidden_size / 2)
        self.dropout = torch.nn.Dropout(drop_p)
 
    def forward(self, q, k, v):
        qk = torch.matmul(query, key.transpose(-2, -1))
        qk = qk.mul(scale_factor)
        scores = qk.softmax(dim=-1)
        return self.dropout(torch.matmul(scores, value))

# Initializing the model
m = Model()

# Inputs to the model
hidden_size = 4
drop_p = 0.3
scale_factor = math.sqrt(hidden_size / 2)
batch_size = 4
length = 5
q = torch.randn(batch_size, hidden_size, length)
k = torch.randn(batch_size, hidden_size, length)
v = torch.randn(batch_size, hidden_size, length)
