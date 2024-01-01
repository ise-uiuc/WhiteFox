
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.n_head = 16
        self.key_size = 1024  # 1024=4x4x64
        self.value_size = 1024  # 1024=4x4x64
        
 
    def forward(self, r_q, r_k, r_v, attn_mask):
        v_q = self.q(r_q).view(1, -1, self.n_head, self.key_size // self.n_head).transpose(1, 2)
        v_k = self.k(r_k).view(1, -1, self.n_head, self.key_size // self.n_head).transpose(1, 2)
        v_v = self.v(r_v).view(1, -1, self.n_head, self.value_size // self.n_head).transpose(1, 2)
        x1 = v_q.size(-1)
        v_q = v_q.squeeze()
        v_k = v_k.squeeze()
        v_v = v_v.squeeze()
        qk = v_q @ v_k.transpose(-2, -1) / math.sqrt(x1)
        qk = qk + attn_mask
        x2 = torch.softmax(qk, dim=-1)
        attn_weight = torch.nn.Softmax(dim=-1)
        output = x2 @ v_v
        return output
 
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.query_size = 24  # 24=4x4x16
        self.key_size = 24  # 24=4x4x16
        self.value_size = 24  # 24=4x4x16
        self.n_head = 16
 
 
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.query = torch.nn.Conv2d(16, 8, 1, stride=1, padding=0)
        self.key = torch.nn.Conv2d(16, 8, 1, stride=1, padding=0)
        self.value = torch.nn.Conv2d(16, 8, 1, stride=1, padding=0)
 
# Initializing the model
m = Model()

# Inputs to the model
r_q = torch.randn(1, 16, 64, 64)
r_k = torch.randn(1, 16, 64, 64)
r_v = torch.randn(1, 16, 64, 64)
attn_mask = torch.randn(8, 8, 1, 1)
