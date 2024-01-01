
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, y1, y2):
        q = self.q
        k = self.k
        v = self.v
        qk = q @ k.transpose(-2, -1) / math.sqrt(q.size(-1))
        qk = qk + y1 # Add attention mask to scaled dot product
        attn_weight = torch.softmax(qk, dim=-1)
        output = attn_weight @ v
        return output

# Initializing the model
m = Model()

# Inputs to the model
y1 = torch.randn(1, 7, 7)
y2 = torch.zeros(1, 7, 7)
y2[0][0][0] = 10000.
m.q = torch.nn.Linear(7, 7)
m.k = torch.nn.Linear(7, 7)
m.v = torch.nn.Linear(7, 7)
