
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.query = torch.nn.Linear(4, 4)
        self.key = torch.nn.Linear(3, 3)
        self.value = torch.nn.Linear(5, 5)
 
    def forward(self, input_tensor, attn_mask):
        t1 = self.query(input_tensor)
        t2 = self.key(input_tensor)
        qk = t1 @ t2.transpose(-2, -1) / math.sqrt(t1.size(-1))
        qk = qk + attn_mask
        attn_weight = torch.softmax(qk, dim=-1)
        output = attn_weight @ self.value(input_tensor)
        return output

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 3, 5, 4)
x2 = torch.randn(1, 3, 5, 3)
x3 = torch.randn(1, 3, 5, 5)
attn_mask = torch.randint(0, 2, size=[1, 1, 5, 5], dtype=torch.long)
