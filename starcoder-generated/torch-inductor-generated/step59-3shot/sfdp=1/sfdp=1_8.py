
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, x1, weight1, bias1, x2, weight2, bias2):
        v1 = torch.matmul(x1, weight1.transpose(0, 3))
        v2 = v1[:,:, ::2, ::2]
        v3 = torch.matmul(v2, weight2.transpose(0, 3))
        v4 = v3 + bias2
        v5 = torch.matmul(v4, weight2)
        v6 = v5[:,:, ::2, ::2]
        v7 = v4 + v6
        v8 = torch.sigmoid(v7)
        v9 = v7 + bias1
        v10 = torch.matmul(v9, weight1)
        v11 = v10 + bias1
        v12 = v8 * v11
        return v12

# Initializing the model
m = Model()

# Inputs to the model
weight1 = torch.randn(1024, 1024)
bias1 = torch.randn(1024)
x1 = torch.randn(64, 1024, 8, 8)
weight2 = torch.randn(512, 1024)
bias2 = torch.randn(512)
