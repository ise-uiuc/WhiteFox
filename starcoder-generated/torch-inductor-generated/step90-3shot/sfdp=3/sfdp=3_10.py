
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.key = torch.nn.Parameter(torch.randn(8, 32, 1000))
        self.scale = torch.nn.Parameter(torch.scalar_tensor(10000))
        self.drop = torch.nn.Dropout(drop_p)
 
    def forward(self, query):
        k = self.key
        q = query
        s = self.scale.item()
        softmax_qk = torch.softmax(s * torch.matmul(q, k.transpose(-2, -1)), dim=-1)
        output = self.drop(softmax_qk)
        v = value
        return torch.matmul(output, v)

# Initializing the model
m = Model()

# Inputs to the model
query = torch.randn(4, 8, 32)
value = torch.randn(16, 8, 32)
