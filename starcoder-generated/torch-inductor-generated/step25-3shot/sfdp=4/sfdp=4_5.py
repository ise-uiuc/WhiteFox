
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.query = torch.nn.Linear(128, 128)
        self.key = torch.nn.Linear(128, 128)
        self.value = torch.nn.Linear(128, 128)
 
    def forward(self, x1, x2, x3):
        qk = self.query(x1) @ self.key(x2).transpose(-2, -1)
        qk = self.qk + x3
        attn_weight = torch.softmax(qk, dim=-1)
        output = attn_weight @ self.value(x1)
        return output

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 128)
x2 = torch.randn(1, 128)
x3 = torch.randn(1, 1, 128)
