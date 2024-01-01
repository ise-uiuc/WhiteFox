
class Model(torch.nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        n_head = 4
        self.h = 8
        self.d = hidden_size
        self.query = torch.nn.Linear(hidden_size, hidden_size)
        self.key = torch.nn.Linear(hidden_size, hidden_size)
        self.value = torch.nn.Linear(hidden_size, hidden_size)

    def forward(self, q, k, v):
        attn_mask = torch.zeros(k.size()[0], 1, k.size()[1], k.size()[1], dtype=torch.float32) 
        qk = self.query(q)
        qk = qk @ k.transpose(-2, -1)
        qk = qk / np.sqrt(self.d)
        attn_mask = torch.zeros(k.size()[0], 4, k.size()[1], k.size()[1], dtype=torch.float32) 
        qk = qk + attn_mask
        attn_weight = torch.softmax(qk)
        output = value @ attn_weight

# Initializing the model
hidden_size = 512
m = Model(hidden_size)

# Inputs to the model
q = torch.randn(1, m.h, hidden_size)
k = torch.randn(1, m.h, hidden_size)
v = torch.randn(1, m.h, hidden_size)
