
class Model(torch.nn.Module):
    def __init__(self, key_dim, num_heads, dropout_p=0.1):
        super(Model, self).__init__()
        self.key_dim = key_dim
        self.num_heads = num_heads
        self.dropout_p = dropout_p
        self.qkv = torch.nn.Conv2d(3, 3 * 3 * num_heads, 1, stride=1, padding=1)
        self.o = torch.nn.Conv2d(3 * num_heads, 3, 1, stride=1, padding=1)
 
        self.scale_factor = math.sqrt(key_dim)
 
    def forward(self, x):
        x = self.qkv(x)
        query, key, value = x.split([3 * self.num_heads, 3 * self.num_heads, 3 * self.num_heads], dim=1)
        qk = torch.matmul(query, key.transpose(-2, -1))
        scaled_qk = qk.div(self.scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=self.dropout_p)
        output = torch.matmul(dropout_qk, value)
        return self.o(output.view(-1, 3 * self.num_heads, x.shape[2], x.shape[3])).view(-1, 3, x.shape[2], x.shape[3]) 

# Initializing the model
m = Model(key_dim=128,
          num_heads=4)

# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
