
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.scale_factor = nn.Parameter(torch.randn([1, 8]))
        self.dropout = nn.Dropout(self.dropout_p)
 
    def forward(self, x1, x2, x3, x4, x5):
        v2 = x1.matmul(x2.transpose(-2, -1))
        v3 = v2.mul(self.scale_factor)
        v4 = F.softmax(v3, -1)
        v5 = self.dropout(v4)
        v6 = v5.matmul(x3)
        v7 = v6.matmul(x4)
        v8 = v7.matmul(x5)
        return v8

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn([1, 8, 64])
x2 = torch.randn([1, 4, 32])
x3 = torch.randn([1, 4, 64])
x4 = torch.randn([1, 4, 32])
x5 = torch.randn([1, 4, 8])
