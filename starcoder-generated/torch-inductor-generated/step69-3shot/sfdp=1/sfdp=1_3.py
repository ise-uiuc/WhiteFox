
class Model(torch.nn.Module):
    def __init__(self, num_heads, hidden_dim, dropout_p):
        super(Model, self).__init__()
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.dropout_p = dropout_p
        self.qkv = torch.nn.Linear(hidden_dim, 3 * hidden_dim)
        self.dropout = torch.nn.Dropout(dropout_p)
        self.attention_weight = torch.nn.Parameter(torch.FloatTensor(1, num_heads, 1, 1))
 
    def forward(self, x):
        qkv = self.qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda x: x.contiguous().view(-1, self.num_heads, self.hidden_dim).transpose(1, 2), qkv)
        qkv_t= torch.matmul(q, k.transpose(-2, -1))
        qkv_scaled= qkv_t.div(self.attention_weight)
        attention_weight_softmax= qkv_scaled.softmax(dim=-1)
        attention_weight_dropout= torch.dropout(attention_weight_softmax, p=self.dropout_p).dropout
        output = torch.matmul(attention_weight_dropout, v)
        output_transposed= output.transpose(1, 2).contiguous().view(x.size(0), x.size(1), self.hidden_dim)
        return output_transposed

# Initializing the model
m = Model(num_heads=8, hidden_dim=64, dropout_p=0.5)

# Inputs to the model
x= torch.randn(128, 256, 64)
__output__= m(x)

