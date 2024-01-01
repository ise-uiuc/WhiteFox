
class Model(torch.nn.Module):
    def __init__(self, hidden_size, num_heads):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.query = torch.nn.ModuleList([torch.nn.Linear(hidden_size, hidden_size) for _ in range(num_heads)])
        self.key = torch.nn.ModuleList([torch.nn.Linear(hidden_size, hidden_size) for _ in range(num_heads)])
        self.value = torch.nn.ModuleList([torch.nn.Linear(hidden_size, hidden_size) for _ in range(num_heads)])

    def forward(self, x1, x2, x3, x4, x5):
        p1 = [t(x1) for t in self.query]
        p2 = [t(x2) for t in self.key]
        p3 = [t(x3) for t in self.value]
        query = torch.stack(p1).transpose(0, 1)
        key = torch.stack(p2).transpose(0, 1)
        value = torch.stack(p3).transpose(0, 1)

        p4 = [torch.matmul(query[i], key[i].transpose(-2, -1)) for i in range(self.num_heads)]
        inv_scale_factor = torch.rsqrt(torch.tensor(self.hidden_size, dtype=torch.float))
        p5 = [item.div(inv_scale_factor) for item in p4]
        softmax_qk = [torch.nn.functional.softmax(p5[i], dim=-1) for i in range(self.num_heads)]
        p7 = [torch.nn.functional.dropout(softmax_qk[i], p=0.0) for i in range(self.num_heads)]
        output = [torch.matmul(p7[i], value[i]) for i in range(self.num_heads)]
        v1 = torch.stack(output).permute(1, 0, 2, 3)
        return v1

# Initializing the model
hidden_size = 128
num_heads = 3
m = Model(hidden_size, num_heads)

# Inputs to the model
x1 = torch.randn(32, 64, hidden_size)
x2 = torch.randn(32, 1, hidden_size)
x3 = torch.randn(32, hidden_size * 2, hidden_size)
x4 = torch.randn(32, 1, hidden_size)
x5 = torch.randn(32, 1 + 1, hidden_size)
