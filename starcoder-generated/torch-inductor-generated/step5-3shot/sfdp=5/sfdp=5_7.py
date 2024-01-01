
class Model(torch.nn.Module):
    def __init__(self, numheads, hidden_size, dropout_p):
        super().__init__()
        self.wq = torch.nn.Linear(hidden_size, hidden_size)
        self.wk = torch.nn.Linear(hidden_size, hidden_size)
        self.wv = torch.nn.Linear(hidden_size, hidden_size)
        self.dense1 = torch.nn.Linear(15, 4)
    
    def forward(self, q, k, v, attn_mask):
        wq = self.wq(q)
        wk = self.wk(k)
        wv = self.wv(v)
        qk = wq @ wk.transpose(-2, -1) / math.sqrt(hidden_size)
        qk = qk + attn_mask
        attn_weight = torch.softmax(qk, dim=-1)
        attn_weight = torch.dropout(attn_weight, dropout_p, True)
        output = attn_weight @ wv
        output = self.dense1(output)
        return output
        
# Initializing the model
m = Model(numheads, hidden_size, dropout_p)

# Inputs to the model
x1 = torch.randn(2, 10, hidden_size)
x2 = torch.randn(2, 10, hidden_size)
x3 = torch.randn(2, 10, hidden_size)
x4 = torch.randn(2, 10, 10) * -10000
