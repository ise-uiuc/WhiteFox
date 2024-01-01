
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        num_heads = 8
        head_dim = 64//num_heads
        self.project_q = torch.nn.ModuleList([copy.deepcopy(torch.nn.Linear(32, 128)) for _ in range(num_heads)])
        self.project_v = torch.nn.ModuleList([copy.deepcopy(torch.nn.Linear(64, 128)) for _ in range(num_heads)])
        self.project_k = torch.nn.ModuleList([copy.deepcopy(torch.nn.Linear(64, 128)) for _ in range(num_heads)])
        self.project_o = torch.nn.ModuleList([copy.deepcopy(torch.nn.Linear(128, 32)) for _ in range(num_heads)])
        self.softmax = torch.nn.Softmax(-1)
 
        self.layernorm = torch.nn.LayerNorm(32)
 
    def forward(self, x1, x2, x3, x4, x5, x6):
        qo = [torch.squeeze(p(torch.unsqueeze(x1, axis=0))) for p in self.project_q]
        vo = [torch.squeeze(p(torch.unsqueeze(x2, axis=0))) for p in self.project_v]
        ko = [torch.squeeze(p(torch.unsqueeze(x3, axis=0))) for p in self.project_k]
 
        q = [qi/math.sqrt(qi.shape[-1]) for qi in qo]
        k = [math.exp(ki-torch.max(ki)) for ki in ko]
        v = [vi/math.sqrt(vi.shape[-1]) for vi in vo]
 
        k = [ki/torch.sum(ki) for ki in k]
        o = []
        oo = []
 
        for i in range(len(q)):
            output = torch.matmul(torch.matmul(q[i], k[i]), v[i])
            output = self.project_o[i](output)
            o.append(output)
            oo.append(output)
 
        o = self.layernorm(torch.stack(o).sum(0))
        