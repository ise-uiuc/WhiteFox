
class Model(torch.nn.Module):
    def __init__(self, num_heads=6, input_dim=1024, hidden_dim=8, output_dim=1024):
        super().__init__()
        self.num_heads = num_heads
        self.qk = torch.nn.Linear(input_dim, hidden_dim * num_heads)
        self.v = torch.nn.Linear(input_dim, output_dim)
 
    def forward(self, x1, x2, x3):
        qk = self.qk(x1)
        split_qk = qk.split(self.num_heads, dim=-1)
        query = split_qk[0].reshape(-1, 1, self.hidden_dim)
        key = split_qk[1].reshape(-1, 1, self.hidden_dim)
        scale_factor = torch.sqrt(torch.tensor(float(self.hidden_dim)))
        scaled_qk = torch.matmul(query, key.transpose(-2, -1)) * scale_factor
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=0.5)
        v = self.v(x2)
        o = torch.matmul(dropout_qk, v)
        return o

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 10, 64)
x2 = torch.randn(1, 25, 64)
x3 = torch.randn(1, 25, 64)
