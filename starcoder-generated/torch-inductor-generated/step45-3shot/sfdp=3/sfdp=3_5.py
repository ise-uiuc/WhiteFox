
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.softmax = torch.nn.Softmax(dim=-1)
        self.dropout = torch.nn.Dropout(p=0.1)
 
    def forward(self, query, key, value, scale_factor, dropout_p):
        v0 = torch.matmul(query, key.transpose(-2, -1))
        v1 = v0 * torch.tensor(scale_factor, dtype=torch.float, device=v0.device)
        v2 = v1.softmax(dim=-1)
        v3 = self.dropout(v2)
        v4 = torch.matmul(v3, value)
        return v4

# Initializing the model
m = Model()

# Inputs to the model
query = torch.randn(1, 8, 256)
key = torch.randn(1, 8, 256)
value = torch.randn(1, 8, 256)
__scale_factor__ = 10
__dropout_p__ = 0.1
