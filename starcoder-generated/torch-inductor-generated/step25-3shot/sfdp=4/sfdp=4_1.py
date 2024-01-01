
class Model(torch.nn.Module):
    def __init__(self, nhead, nhid, dropout):
        super().__init__()
        self.nhead = nhead
        self.nhid = nhid
        self.dropout = dropout
        self.h = torch.nn.ModuleList([torch.nn.Linear(nhid, nhid) for _ in range(nhead)])
        self.attn_dropout = torch.nn.Dropout(dropout)
        self.o = torch.nn.Linear(nhid, nhid)
 
    def forward(self, x1, x2, x3):
        bs = x1.size(0)
        x4 = torch.empty(bs, self.nhead, self.nhid, device=x1.device)
        for i in range(self.nhead):
            # Compute query, key and value from input tensor x2
            head = self.h[i](x2).view(bs, -1, self.nhid // self.nhead)
            q = head @ head.transpose(-2, -1)
            k = head @ head.transpose(-2, -1)
            v = head @ head.transpose(-2, -1)
            w = q @ k.transpose(-2, -1) / math.sqrt(q.size(-1)) + x3
            a = torch.softmax(w, dim=-1)
            x4[:, i, :] = a @ head
        x = x4.transpose(1, 2).contiguous().view(bs, -1)
        x = self.o(x)
        return x

# Initializing the model
dropout = 0.2
nhead = 2
nhid = 128
m = Model(nhead, nhid, dropout)
 
# Inputs to the model
x1 = torch.randn(5, 24, 128)
x2 = torch.randn(5, 24, 128)
attn_mask = torch.randint(0, 1, (5, nhead * (1 + x2.size(1)), nhead * (1 + x2.size(1))))
if torch.cuda.is_available():
    x1 = x1.to('cuda')
    x2 = x2.to('cuda')
    attn_mask = attn_mask.to('cuda')
