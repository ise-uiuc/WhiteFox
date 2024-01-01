
class Attention(torch.nn.Module):
    def __init__(self, hidden_size, dropout_p=0.2):
        super().__init__()

        self.w_q = torch.nn.Linear(hidden_size, hidden_size)
        self.w_k = torch.nn.Linear(hidden_size, hidden_size)
        self.w_v = torch.nn.Linear(hidden_size, hidden_size)
        self.dropout = torch.nn.Dropout(dropout_p)

    def forward(self, q, k, v, mask):
        q_, k_, v_ = self.w_q(q), self.w_k(k), self.w_v(v)
        output = self.dropout(torch.matmul(q_, k_.transpose(0, 1)) / math.sqrt(k_.size(-1)))
        if mask is not None:
            output.masked_fill_(mask.unsqueeze(1) == 0, -1e9)
        return output

# Initializing the model
m = Attention(hidden_size=16)

# Inputs to the model
q = torch.randn(16, 32, 16)
k = torch.randn(16, 32, 16)
v = torch.randn(16, 32, 16)
mask = torch.where(torch.arange(q.size(1)).to(q.device) < 16, torch.tensor([1]).to(q.device), torch.tensor([0]).to(q.device))
q = q.transpose(0, 1)
k = k.transpose(0, 1)
v = v.transpose(0, 1)
mask = mask.transpose(0, 1)
