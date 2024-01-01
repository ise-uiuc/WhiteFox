
class Layer(torch.nn.Module):
    def __init__(self, num_heads, dim_model, dim_head):
        super(Layer, self).__init__()
        self.num_heads = num_heads
        self.dim_head = dim_head
        self.dim_model = dim_model
        self.scale_factor = dim_head ** -0.5
        
        self.q = torch.nn.Linear(dim_model, num_heads * dim_head, bias=False)
        self.k = torch.nn.Linear(dim_model, num_heads * dim_head, bias=False)
        self.v = torch.nn.Linear(dim_model, num_heads * dim_head, bias=False)
        self.fc = nn.Linear(num_heads * dim_head, dim_model)
        self.dropout = torch.nn.Dropout()
    
    def forward(self, x, mask, attn_bias):
        q = self.q(x)
        k = self.k(x)
        v = self.v(x)
        q, k, v = self.split_head(q, k, v)
        
        dot_product = torch.matmul(q, k.transpose(-2, -1)) * self.scale_factor
        dot_product += attn_bias.squeeze(1)
        dot_product = torch.nn.functional.softmax(dot_product, dim=-1)
        dot_product = self.dropout(dot_product)
        
        context = torch.matmul(dot_product, v)
        context = self.combine_head(context)
        return self.fc(context)

# Initializing the model
m = Layer(num_heads=24, dim_model=64, dim_head=16)

# Inputs to the model
x = torch.randn(1, 298, 64)
mask = torch.randint(2, size=(1, 24, 99, 99)) # a float tensor
attn_bias = torch.randn(1, 24, 99, 99)
