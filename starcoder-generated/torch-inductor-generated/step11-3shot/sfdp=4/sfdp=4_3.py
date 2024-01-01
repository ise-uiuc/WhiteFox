
class Model(torch.nn.Module):
    def __init__(self, n, nhead):
        super().__init__()
        self.w_q = nn.Linear(1, 1, bias=False)
        self.w_kv = nn.Linear(1, nhead, bias=False)
        self.w_o = nn.Linear(nhead + 1, 1)
        self.n = n
        self.nhead = nhead

    def forward(self, q, v, attn_mask):    
        q_ = q * 1
        kv = torch.rand(self.n, self.n) * 1
        q_kv = torch.cat((q_,kv), 1)
        q_ = self.w_q(q_)
        
        kv = self.w_kv(kv)
        kv = kv.view(-1, self.nhead, self.n // self.nhead, 1)
        kv = kv.permute(0, 2, 3, 1)
        
        q_kv = self.w_o(q_kv)
        q_kv = q_kv.view(-1, self.nhead, self.n // self.nhead, 1)
        q_kv = q_kv.permute(0, 2, 3, 1)

        q_ = q_.permute(2, 0, 1).unsqueeze(0)
        attn_weight_ = torch.flatten(torch.matmul(q_, kv), 1)
        attn_weight_ = attn_weight_ * 1
        attn_weight = attn_weight_ * attn_mask
        attn_weight = self.softmax(attn_weight, 3)
        output = torch.matmul(attn_weight, q_kv)
        _, n_head, h, _ = output.shape
        output = torch.reshape(output, (h, 1, n_head * 1))
        output = self.w_o(output)

        return output
      
# Initializing parameters
n = 10
nhead = 10
batch_size = 1
attn_mask = torch.ones((nhead, nhead))

# Construct a dummy input tensor
q = torch.rand(batch_size, nhead, n)
v = torch.rand(batch_size, nhead, n)

# Inputs to the model
