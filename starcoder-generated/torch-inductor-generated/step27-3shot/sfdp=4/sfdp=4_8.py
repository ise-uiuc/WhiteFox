
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.attention_dropout = 0.1
        self.attention_mask = torch.tensor([[ [[0,0], [0,0]]]], dtype=torch.float)
        self.softmax = torch.nn.Softmax(dim=-1)
        self.matmul1 = torch.nn.Linear(288, 288, bias=False)
        self.matmul2 = torch.nn.Linear(288, 288, bias=False)
        self.matmul3 = torch.nn.Linear(288, 288, bias=False)
 
    def forward(self, x):
        v0 = self.matmul1(x)
        v1 = self.matmul2(x)
        qk = v0 @ v1.transpose(-2, -1)
        qk = qk / math.sqrt(math.sqrt(qk.size(-1)))
        qk = qk + self.attention_mask
        attn_weight = self.softmax(qk)
        v3 = self.matmul3(x)
        v4 = attn_weight @ v3
        v5 = v4 * v2
        return v5

# Initializing the model
m = Model()

# Inputs to the model
x = torch.randn(1, 288, dtype=torch.float)
v2 = self.attention_dropout
