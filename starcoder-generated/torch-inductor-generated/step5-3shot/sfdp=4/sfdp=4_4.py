
class Model(torch.nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.query_key_value = torch.nn.Linear(hidden_size, hidden_size * 3)
 
    def forward(self, x1, x2):
        qkvpq = self.query_key_value(x1)
        if x2.dim() == 3:
            x2 = x2.unsqueeze(0)
        if x2.dim() == 4:
            x2 = x2.view(x2.size(0), x2.size(1), 1, self.hidden_size).transpose(1, 2)
        qkvpq = qkvpq.view(qkvpq.size(0), qkvpq.size(1), 3, self.hidden_size)
        q, k, v = qkvpq.chunk(3, dim=-2)
        
        v2 = q @ k.transpose(-2, -1)
        v2 = v2 / math.sqrt(q.size(-1))
        if x2.size()!= v2.size():
            x2 = x2.expand(v2.size())
        v2 = v2 + x2
        
        v3 = torch.softmax(v2, dim=-1)
        x3 = v3 @ v
        return x3


# Initializing the model
m = Model(hidden_size=16)

# Inputs to the model
x1 = torch.randn(7, 24, 16)
x2 = torch.randn(7, 8, 16)
