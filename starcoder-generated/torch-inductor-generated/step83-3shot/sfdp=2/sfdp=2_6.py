
class Model(torch.nn.Module):
    def __init__(self, num_attention_heads, dim_model, dim_key, dim_value, dropout_p):
        super().__init__()
        self.dot_product_projection_q = torch.nn.Linear(dim_model, num_attention_heads * dim_key)
        self.dot_product_projection_k = torch.nn.Linear(dim_model, num_attention_heads * dim_key)
        self.dot_product_projection_v = torch.nn.Linear(dim_model, num_attention_heads * dim_value)
        self.dropout_p = dropout_p
        self.scale_factor = dim_key ** -0.5
 
        self.output_matrix_dropout = torch.nn.Dropout(dropout_p)
        self.output_projection = torch.nn.Linear(num_attention_heads * dim_value, dim_model)
 
    def forward(self, x1, x2, x3):
        q = self.dot_product_projection_q(x1)
        k = self.dot_product_projection_k(x2)
        v = self.dot_product_projection_v(x3)
 
        s = torch.matmul(q.transpose(-2, -1), k)
 
        s.div_(self.scale_factor)
 
        m = torch.nn.functional.softmax(s, dim=-1)
 
        m.mul_(self.dropout_p)
 
        m = torch.nn.functional.dropout(m, p=self.dropout_p)
 
        out = torch.matmul(m, v)
 
        out = self.output_matrix_dropout(out)
 
        out = self.output_projection(out)
 
        return out

# Initializing the model
m = Model(16, 128, 128, 128, 0.2)

# Inputs to the model
x1 = torch.randn(1, 15, 128) # The size of the query
x2 = torch.randn(1, 14, 128) # The size of the key
x3 = torch.randn(1, 17, 128) # The size of the value
