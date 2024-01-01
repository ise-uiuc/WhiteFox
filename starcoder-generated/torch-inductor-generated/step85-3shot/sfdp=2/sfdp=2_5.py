
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.nhead = 8
        self.batch_size = 1
        self.dim = 16
        self.dim_head = self.dim // self.nhead
        self.fc_query = torch.nn.Linear(self.dim, self.dim, bias=False)
        self.fc_key = torch.nn.Linear(self.dim, self.dim, bias=False)
        self.dropout = torch.nn.Dropout(0.1)

    def forward(self, q, kv):
        v_dim = kv.shape[-1]
        q = self.fc_query(q).view(
            self.batch_size,
            self.nhead,
            self.dim_head
        ).transpose(0, 1)
        key = self.fc_key(kv).view(
            1,
            self.nhead,
            self.dim_head
        ).transpose(0, 1)
        kv /= v_dim**0.5
        q *= v_dim**0.5
        q_flat = q.transpose(1, 2).contiguous().view(1, self.dim_head, -1)
        k_flat = key.transpose(1, 2).contiguous().view(1, self.dim_head, -1)
        sim = torch.matmul(q_flat, k_flat.transpose(1, 2)) 
        softmax_sim = torch.nn.functional.softmax(sim, dim=-1) 
        drop_sim = torch.nn.functional.dropout(softmax_sim, p=0.03) 
        output = torch.matmul(drop_sim, kv.transpose(0, 1).view(v_dim, v_dim)) 
        return output 

# Initializing the model
m = Model()

# Inputs to the model
q = torch.randint(255, size=(8, 12))
