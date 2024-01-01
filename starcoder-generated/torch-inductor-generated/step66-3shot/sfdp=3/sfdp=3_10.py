
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.query = torch.nn.Parameter(torch.randn(8, 8, 256, 256))
        self.key = torch.nn.Parameter(torch.randn(8, 8, 256, 256))
        self.value = torch.nn.Parameter(torch.randn(8, 8, 256, 256))
        self.fc = torch.nn.Linear(256, 256)
        self.dropout = torch.nn.Dropout(p=0.0)
 
    def forward(self, x1, x2):
        v1 = torch.einsum('bctv,bd->bvtd', x1, self.query)
        v2 = torch.einsum('bctv,bd->bvtd', x2, self.key)
        v3 = torch.matmul(v1, v2.transpose(-2, -1))
        v4 = v3 * 1
        v5 = v3.softmax(dim=-1)
        v6 = self.dropout(v5)
        v7 = torch.matmul(v6, self.value)
        v8 = v7.permute(1,2,0,3)
        v9 = v8.contiguous().view(1,256, 32, 32)
        v10 = self.fc(v9)
        return v10

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 8, 256, 256)
x2 = torch.randn(32, 8, 256, 256)
