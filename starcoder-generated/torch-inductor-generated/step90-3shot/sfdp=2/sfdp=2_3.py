
class Model(torch.nn.Module):
    def compute_attention(self, q, k, v, dropout_p):
        inv_scale_factor = math.sqrt(v.size(-1))
        dot_prod = torch.matmul(q, k.transpose(-2, -1))
        scaled_dot_prod = dot_prod.div(inv_scale_factor)
        softmax_result = scaled_dot_prod.softmax(dim=-1)
        dropout_result = torch.nn.functional.dropout(softmax_result, p=dropout_p)
        output = dropout_result.matmul(v)
        return output 
 
    def forward(self, queries, keys, values, dropout_p):
        query_1 = self.compute_attention(queries[:, 0], keys[:, 0], values[:, 0], dropout_p)
        query_2 = self.compute_attention(queries[:, 1], keys[:, 1], values[:, 1], dropout_p)
        query_3 = self.compute_attention(queries[:, 2], keys[:, 2], values[:, 2], dropout_p)
        output = torch.stack([query_1, query_2, query_3], dim=1)
        return output
    
# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 6, 10, 20)
x2 = torch.randn(1, 6, 10, 20)
x3 = torch.randn(1, 6, 10, 20)
dropout_p = 0.1
