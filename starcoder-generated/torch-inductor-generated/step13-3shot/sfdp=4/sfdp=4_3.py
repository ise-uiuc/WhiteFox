
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.query = torch.nn.Parameter(torch.empty(20, 10).uniform_(-0.1, 0.1))
        self.key = torch.nn.Parameter(torch.empty(20, 20).uniform_(-0.1, 0.1))
        self.value = torch.nn.Parameter(torch.empty(20, 10).uniform_(-0.1, 0.1))
 
    def forward(self, x1):
        q = self.query
        k = self.key
        v = self.value
        qk = q @ k.transpose(-2, -1) / math.sqrt(q.size(-1))
        if x1 is not None:
            attn_mask = (x1 == 0).float().unsqueeze(1) \
              .unsqueeze(1) \
              .expand(x1.size(0), 1, x1.size(1), x1.size(1))
            qk = qk + attn_mask
        attn_weight = torch.softmax(qk, dim=-1)
        output = attn_weight @ v
        return output

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.arange(20).unsqueeze(0).unsqueeze(1).expand(1, 1, 20)

x1 = torch.gt(x1, 15)
