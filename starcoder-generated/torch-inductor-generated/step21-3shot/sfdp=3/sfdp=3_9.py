
class Model(nn.Module):
    def __init__(self, nfeat, nhid, nconv, scale_factor, dropout_p):
        super().__init__()
 
        self.convs = nn.ModuleList()
        for i in range(nconv):
            vq = torch.nn.Conv2d(nhid, nhid, 1, stride=1, padding=0)
            vk = torch.nn.Conv2d(nhid, nhid, 1, stride=1, padding=0)
            vv = torch.nn.Conv2d(nhid, nhid, 1, stride=1, padding=0)
            self.convs.append(torch.nn.ModuleList([vq, vk, vv]))
 
    def forward(self, x, adj):
        outputs = []
        for vq, vk, vv in self.convs:
            q = vq(torch.tanh(x))
            k = vk(torch.tanh(x))
            v = vv(torch.tanh(x))
 
            q = q.permute(0, 2, 3, 1).view(-1, x.size(1)).contiguous()
            k = k.permute(0, 2, 3, 1).view(-1, x.size(1)).contiguous()
            v = v.view(-1, x.size(1)).contiguous()
            adj_t = adj.view(-1, adj.size(1)).contiguous()
 
            q_t = torch.matmul(q, adj_t)
            k_t = torch.matmul(k, x.data.transpose(1, 2)).contiguous()
            logits = torch.bmm(q_t, k_t).view(-1, adj.size(1), x.size(1))
 
            # Attention weights matrix
            a = logits.mul(scale_factor).softmax(dim=-2)
 
            # Output of attention
            o = torch.matmul(a, v).view(q.size(0), q.size(1), adj.size(1), x.size(1))
 
            # Skip connection
            x = x + o
            outputs.append(x)
 
        return x

# Initializing the model
m = Model()

# Inputs to the model
x = torch.randn(15, 5, 8, 8)
adj = torch.randn(15, 10, 5)
