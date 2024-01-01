
class Model(torch.nn.Module):
    def __init__(self, d, n, h):
        super().__init__()

        self.n = n
        self.head_dims = h
        self.scale = self.head_dims ** (-0.5)
        self.dropout_p = 0.1
        self.fc_q = torch.nn.Linear(d, h * n)
        self.fc_k = torch.nn.Linear(d, h * n)
        self.fc_v = torch.nn.Linear(d, h * n)
        self.fc_o = torch.nn.Linear(d, h * n)
 
    def attention(self, q, k, mask=None):
        raw_weight = q @ k.transpose(-2, -1)
        weight = raw_weight / self.scale
        if mask is not None:
            weight.data.masked_fill_(mask.byte(), -float('inf'))
        
        attn = torch.nn.functional.softmax(weight, dim=-1)
        attn = torch.nn.functional.dropout(attn, self.dropout_p, True)
        return attn
 
    def forward(self, x1):
        q = self.fc_q(x1)
        k = self.fc_k(x1)
        v = self.fc_v(x1)
 
        b, d, W = q.size()
        q = q.reshape(b, self.n, self.head_dims, W).transpose(1, 2)
        k = k.reshape(b, self.n, self.head_dims, W).transpose(1, 2)
        v = v.reshape(b, self.n, self.head_dims, W).transpose(1, 2)
        if x1.is_cuda:
            attn_mask = self.create_mask(b, d, W, device='cuda')
        else:
            attn_mask = self.create_mask(b, d, W)
        attn = self.attention(q, k, attn_mask.unsqueeze(-1).unsqueeze(-1))
        output = (attn @ v).transpose(1, 2).reshape(b, d, W)
        output = self.fc_o(output)
        return output
 
    def create_mask(self, b, d, w, device='cpu'):
        if w % self.n!= 0:
            w = (w // self.n + 1) * self.n

        mask = torch.ones(b, 1, d, w)
        return (mask == 1) - 1


# Initializing the model
m = Model(d=1024, n=8, h=64)

# Inputs to the model
x1 = torch.randn(1, 1024, 48)
