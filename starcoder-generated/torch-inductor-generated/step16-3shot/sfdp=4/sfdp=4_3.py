 inputs and outputs
class Model(torch.nn.Module):
    def __init__(self, num_heads, head_size, num_outputs=2048):
        super().__init__()
        self.num_heads = num_heads
        self.head_size = head_size
        self.num_outputs = num_outputs
        
        self.q_fc = torch.nn.Linear(num_outputs, num_heads * head_size)
        self.k_fc = torch.nn.Linear(num_outputs, num_heads * head_size)
        self.v_fc = torch.nn.Linear(num_outputs, num_heads * head_size)
        self.out_fc = torch.nn.Linear(num_heads * head_size, num_outputs)
       
    def forward(self, x1, x2, x3, attn_mask):
        q = self.q_fc(x1)  # [B, num_heads*head_size]
        k = self.k_fc(x2)  # [B, num_heads*head_size]
        v = self.v_fc(x3)  # [B, num_heads*head_size]
        q, k, v = [reshape(x, (-1, self.num_heads, self.head_size)) for x in [q, k, v]]
        q = q.transpose(1, 2)  # (B, head_size, num_heads)
        k = k.transpose(1, 2)  # (B, head_size, num_heads)
        attn = q @ k.transpose(-2, -1) / math.sqrt(q.size(-1))  # (B, head_size, num_heads) x (B, head_size, num_heads) -> (B, num_heads, head_size, head_size) -> (B, num_heads, head_size, num_heads)
        attn = attn + attn_mask
        attn = torch.softmax(attn, dim=-1)
        value = attn @ v  # (B, num_heads, head_size, num_heads) x (B, num_heads, num_heads, head_size) -> (B, num_heads, head_size, head_size)
        value = value.transpose(2, 1).contiguous()  # [B, num_heads, head_size, head_size] -> [B, num_heads, head_size, head_size]
        output = value.view(value.size(0), -1)  # [B, num_heads, head_size, head_size] -> [B, num_heads*head_size]
        return self.out_fc(output)

# Initializing the model
m = Model(num_heads=8, head_size=64)

# Inputs to the model
x1 = torch.randn(1, 512, 2048)  # query tensor
x2 = torch.randn(1, 512, 2048)  # key tensor
x3 = torch.randn(1, 512, 2048)  # value tensor
attn_mask = torch.full((1, 512, 512), -1e9, dtype=torch.float32)
attn_mask = torch.tril(attn_mask)
