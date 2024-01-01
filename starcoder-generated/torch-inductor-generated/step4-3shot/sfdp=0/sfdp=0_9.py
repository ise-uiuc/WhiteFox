
class SelfAttention(torch.nn.Module):
    def __init__(self, heads=1, depth=128):
        super().__init__()
        self.depth = depth
        self.heads = heads
        self.wq = torch.nn.Linear(depth, depth) # The linear layer that computes `query`.
        self.wk = torch.nn.Linear(depth, depth) # The linear layer that computes `key`.
        self.wv = torch.nn.Linear(depth, depth) # The linear layer that computes `value`.
        self.wo = torch.nn.Linear(depth, depth) # The linear layer that computes the output.
 
    def forward(self, x1, x2):
        v1 = self.wq(x1)
        v2 = self.wk(x2)
        v3 = self.wv(x2)
        v4 = torch.matmul(v1, v2.transpose(2, 3))/math.sqrt(self.depth/self.heads)
        v5 = v4.softmax(-1)
        v6 = torch.matmul(v5, v3)
        v7 = v6.split(self.heads, -1)
        v8 = torch.cat(v7, 0)
        v9 = v8.transpose(1, 2)
        v10 = v9.split(10, 0)
        v11 = torch.cat(v10, 1)
        v12 = self.wo(v11)
        return v12
        
# Initializing the model
m = SelfAttention()

# Inputs to the model
x1 = torch.randn(10, 32, 128)
x2 = torch.randn(10, 64, 128)
