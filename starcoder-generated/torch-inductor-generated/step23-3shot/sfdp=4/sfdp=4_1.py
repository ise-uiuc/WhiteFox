
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dense1 = torch.nn.Linear(4, 20)
        self.q_linear = torch.nn.Linear(4, 20)
        self.k_linear = torch.nn.Linear(4, 20)
        self.v_linear = torch.nn.Linear(4, 20)
        
    def forward(self, query, key, value, mask):
        b_s, nq = query.shape[:2]
        nk = key.shape[2]
        nv = value.shape[2]
        q = torch.reshape(self.q_linear(query), [b_s, nq, 1, 20])
        k = torch.reshape(self.k_linear(key), [b_s, 1, nk, 20])
        v = torch.reshape(self.v_linear(value), [b_s, 1, nv, 20])
        
        dots = torch.matmul(q, k)
        dots = dots / math.sqrt(20)
        
        if mask is not None:
            mask_value = -1e9
            dots = torch.where(mask==0, dots, mask_value)
        
        attn_weight = torch.softmax(dots, dim=-1)
        output = torch.matmul(attn_weight, v)

        return output

# Initializing the model
m = Model()

# Inputs to the model
batch_size = 4
query = torch.randn(batch_size, 22, 4)
key = torch.randn(batch_size, 50, 4)
value = torch.randn(batch_size, 50, 4)
mask = torch.ones([batch_size, 22, 50], dtype=torch.uint8)
mask[0][0][3] = 0
mask[1][3][10] = 0
mask[2][22][5] = 0
mask[3][47][1] = 0
