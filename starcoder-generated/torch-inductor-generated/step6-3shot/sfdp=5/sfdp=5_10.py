
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.q_linear = torch.nn.Linear(64, 64)
        self.k_linear = torch.nn.Linear(64, 64)
        self.v_linear = torch.nn.Linear(64, 64)
        self.out_linear = torch.nn.Linear(64, 64)
 
    def forward(self, query, key, value, attn_mask):
        q = self.q_linear(query)
        k = self.k_linear(key)
        v = self.v_linear(value)
        attn_weight = (q @ k.transpose(-1, -2)) / math.sqrt(64)
        attn_weight = attn_weight + attn_mask
        attn_weight = torch.softmax(attn_weight, -1)
        attn_weight = torch.dropout(attn_weight, 0.2, training=self.training)
        output = attn_weight @ v
        return self.out_linear(output)
# Random input data
x1 = torch.randn(2, 64)
x2 = torch.randn(2, 64)
x3 = torch.randn(2, 64)
x4 = (torch.rand(2, 2) < 0.05).to(x3.dtype)

# Building and running the model
m = Model()
