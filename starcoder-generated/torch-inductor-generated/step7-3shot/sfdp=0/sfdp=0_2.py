
class Model(torch.nn.Module):
    def __init__(self, num_tokens, dim, num_heads, sequence_length):
        super().__init__()
        self.inv_scale = torch.sqrt(torch.tensor(dim, dtype = torch.float))
        self.num_heads = num_heads
        self.attention_head_size = dim // num_heads
        self.W_q = torch.nn.Linear(dim, dim)
        self.W_k = torch.nn.Linear(dim, dim)
        self.W_v = torch.nn.Linear(dim, dim)
        self.dense = torch.nn.Linear(num_heads * self.attention_head_size, dim)
 
    def forward(self, x4):
        q = self.W_q(x4)
        k = self.W_k(x4)
        v = self.W_v(x4)
        q, k, v = [x.reshape(x.shape[0], x.shape[1], self.num_heads, self.attention_head_size).transpose(1, 2) for x in [q, k, v]]
        scaled_dot_product = torch.matmul(q, k.transpose(-2, -1))
        inv_scale = torch.reshape(self.inv_scale, shape=[1, 1, 1, 1])
        scaled_dot_product = scaled_dot_product / inv_scale
        attention_weights = scaled_dot_product.softmax(dim=-1)
        output = attention_weights.matmul(v)
        output = output.transpose(1, 2).reshape(output.shape[0], output.shape[1], self.num_heads * self.attention_head_size, 1, 1)
        return self.dense(output).squeeze(-2).squeeze(-2)
 
model = Model(num_tokens = 1024, dim=1024, num_heads=64, sequence_length=1024)
input = torch.rand(32, 1024, 1, 1, 1)
output = model(input)

# Inputs to the model
x5 = input
