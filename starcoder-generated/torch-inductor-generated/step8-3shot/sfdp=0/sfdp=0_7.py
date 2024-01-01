
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.input_dim= 128
        self.inter_proj_dim = self.input_dim * 3
        self.num_attention_heads = 4
        self.attention_head_size = self.input_dim // self.num_attention_heads
        self.inv_sqrt_dim = np.power(self.input_dim, -0.5)

    def __call__(self, q, k, v):
        q *= self.inv_sqrt_dim
        k *= self.inv_sqrt_dim

        attn_weights = torch.matmul(q, k)
        scale_weights = F.softmax(attn_weights, dim=-1)
        output = torch.matmul(scale_weights, v)
        return output

# Initializing the model
m = Model()

# Inputs to the model
q = torch.tensor(np.zeros([32, 2, 128]), dtype=torch.float)
k = torch.tensor(np.zeros([32, 1, 128]), dtype=torch.float)
v = torch.tensor(np.zeros([32, 1, 128]), dtype=torch.float)
