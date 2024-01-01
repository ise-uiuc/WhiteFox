
class Model(torch.nn.Module):
    def __init__(self, num_heads, dim_per_head, dropout_p):
        super().__init__()
        self.num_heads = num_heads
        self.dim_per_head = dim_per_head
        self.scale_factor = torch.sqrt(torch.FloatTensor([dim_per_head])).to(torch.device('cuda:0'))
        self.dropout_p = dropout_p
 
    def forward(self, query, key, value):
        qk = torch.matmul(query, key.transpose(-2, -1))
        scaled_qk = qk.div(self.scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=self.dropout_p)
        output = dropout_qk.matmul(value)
        return output

# Initializing the model
m = Model(num_heads=8, dim_per_head=64, dropout_p=0.0)

# Inputs to the model
query = torch.randn(1, 64, 512, 64)
key = torch.randn(1, 64, 1024, 64)
value = torch.randn(1, 64, 1024, 64)
