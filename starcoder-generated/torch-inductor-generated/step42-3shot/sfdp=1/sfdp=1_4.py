
class Model(torch.nn.Module):
    def __init__(self, num_heads):
        super().__init__()
        self.query = torch.nn.Linear(3, 4)
        self.key = torch.nn.Linear(3, 4)
        self.value = torch.nn.Linear(3, 4)
 
        self.num_heads = num_heads
        self.input_len = 3
        self.key_len = 4
        self.value_len = 4
        self.scale_factor = self.key_len ** -0.5

    def forward(self, inp):
        vq = self.query(inp).reshape(self.num_heads, self.input_len // self.num_heads, self.key_len)
        vk = self.key(inp).reshape(self.num_heads, self.key_len, self.key_len)
        vv = self.value(inp).reshape(self.num_heads, self.input_len // self.num_heads, self.value_len)
 
        qk = torch.matmul(vq, vk.transpose(-2, -1))
        dk = self.key_len ** -0.5

        softmax_qk = qk.div(self.scale_factor).softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=0.2)
        output = dropout_qk.matmul(vv)
        return output

# Initializing the model
m = Model(num_heads=4)

# Inputs to the model
x1 = torch.randn(20, 2, 64)
