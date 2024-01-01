
class Model(torch.nn.Module):
    def __init__(self, embed_dim, num_heads, dropout_p=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout_p = dropout_p
        self.head_dim = embed_dim // num_heads
        self.inv_scale_factor = self.head_dim ** -0.5
 
        self.dropout = torch.nn.Dropout(p=dropout_p)
        self.reshape = torch.nn.Linear(embed_dim, embed_dim)
 
    def forward(self, x1, x2):
        scale_factor = torch.sqrt(torch.as_tensor(self.head_dim).float())
        qk = torch.matmul(self.reshape(x1), x2.transpose(-2, -1))
        scaled_qk = qk.div(scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = self.dropout(softmax_qk)
        output = dropout_qk.matmul(x2)
        return output

# Initializing the model
m = Model(2048, 16)

# Inputs to the model
x1 = torch.randn(1, 256, 2048)
x2 = torch.randn(1, 256, 2048)
