
class Model(torch.nn.Module):
    def __init__(self, hidden_size=8, num_of_heads=8, output_size=2):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_of_heads = num_of_heads
        self.sqrt_hd = math.sqrt(hidden_size)

        self.q_linear = torch.nn.Linear(hidden_size, hidden_size * num_of_heads)
        self.k_linear = torch.nn.Linear(hidden_size, hidden_size * num_of_heads)
        self.v_linear = torch.nn.Linear(hidden_size, hidden_size * num_of_heads)
        self.o_linear = torch.nn.Linear(hidden_size, hidden_size)
        self.attn_mask = torch.zeros(
            (num_of_heads, 1, 1, hidden_size), dtype=torch.float16)

    def attention(self, q, k, v):
        q = self.q_linear(q)
        k = self.k_linear(k)
        v = self.v_linear(v)
 
        q = torch.cat(torch.split(q, self.hidden_size, -1), 0)
        k = torch.cat(torch.split(k, self.hidden_size, -1), 0)
        v = torch.cat(torch.split(v, self.hidden_size, -1), 0)
 
        q = q.view(self.num_of_heads, -1, 1, self.hidden_size).transpose(-2, -1)
        k = k.view(self.num_of_heads, -1, 1, self.hidden_size).transpose(-2, -1)
        v = v.view(self.num_of_heads, -1, 1, self.hidden_size).transpose(-2, -1)
 
        qk = q @ k / self.sqrt_hd
        qk = qk + self.attn_mask
        qk_softmax = torch.softmax(qk, dim=-1)
        qkv = qk_softmax @ v

        qkv = qkv.transpose(0, 1).contiguous()
        qkv = qkv.view(
            qkv.size(0), qkv.size(2), -1)
      
        m = self.o_linear(qkv)
        return m

    def forward(self, query, key, value):
        output = self.attention(query, key, value)
        return output

# Initializing the model
m = Model()

# Inputs to the model
q = torch.randn(1, 8, 32)
k = torch.randn(1, 8, 64)
v = torch.randn(1, 8, 64)
