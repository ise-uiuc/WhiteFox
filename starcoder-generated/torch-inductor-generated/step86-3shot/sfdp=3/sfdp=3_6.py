
class ExampleModel(torch.nn.Module):
    def __init__(self, n_heads, dropout_p):
        super().__init__()
        self.n_heads = n_heads
        self.dropout_p = dropout_p
 
        self.projection = torch.nn.Linear(n_heads, n_heads)
 
    def forward(self, query, key, value):
        q = self.projection(query).view(query.size(0), self.n_heads, -1, query.size(2))
        k = self.projection(key).view(key.size(0), self.n_heads, -1, key.size(2))
        v = self.projection(value).view(value.size(0), self.n_heads, -1, value.size(2))
        
        qk = torch.matmul(q, k.transpose(-2, -1))
        scale_factor = 1 / math.sqrt(k.size(-1))
        
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=self.dropout_p)
        
        output = dropout_qk.matmul(v)
        output = output.view(output.size(0), -1, self.n_heads * output.size(3))
        return output
        
        
# Initializing the model
m = ExampleModel(n_heads=8, dropout_p=0.2)

# Inputs to the model
x1 = torch.randn(1, 64, 32)
x2 = torch.randn(1, 32, 32)
x3 = torch.randn(1, 32, 32)
