
class Model(torch.nn.Module):
    def __init__(self, num_heads, d_model, dropout_p):
        super().__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        self.dropout_p = dropout_p
        self.qs = torch.nn.Linear(d_model, d_model)
        self.ks = torch.nn.Linear(d_model, d_model)
        self.vs = torch.nn.Linear(d_model, d_model)
        self.dp = torch.nn.Dropout(p=dropout_p)
 
    def forward(self, x1):
        num_batches = x1.shape[0]
        q = self.qs(x1)
        k = self.ks(x1)
        v = self.vs(x1)
        q = q.view(num_batches, -1, self.num_heads, self.d_model//self.num_heads).transpose(1, 2)
        k = k.view(num_batches, -1, self.num_heads, self.d_model//self.num_heads).transpose(1, 2)
        v = v.view(num_batches, -1, self.num_heads, self.d_model//self.num_heads).transpose(1, 2)
        q = q.contiguous().view(num_batches*self.num_heads, -1, self.d_model//self.num_heads)
        k = k.contiguous().view(num_batches*self.num_heads, -1, self.d_model//self.num_heads)
        v = v.contiguous().view(num_batches*self.num_heads, -1, self.d_model//self.num_heads)
        scaled_qk = torch.matmul(q, k.transpose(-2, -1)) * (1/math.sqrt(k.shape[-1]))
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = self.dp(softmax_qk)
        output = torch.matmul(dropout_qk, v)
        output = output.view(num_batches, self.num_heads, -1, self.d_model//self.num_heads)
        output = output.transpose(1, 2)
        output = output.contiguous().view(output.shape[0], output.shape[1], output.shape[2]*output.shape[3])
        return output

# Initializing the model
m = Model(num_heads=2, d_model=3, dropout_p=0.1)

# Inputs to the model
x1 = torch.randn(2, 2, 3)
