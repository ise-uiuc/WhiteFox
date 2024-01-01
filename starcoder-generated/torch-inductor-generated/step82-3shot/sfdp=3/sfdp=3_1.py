
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.scale_factor = math.sqrt(query.size(-1))
        self.softmax = torch.nn.Softmax(dim=-1)
 
    def forward(self, query, key, value, dropout_p):
        dot_product = query.matmul(key.transpose(-2, -1))
        scaled_dot_product = self.scale_factor * dot_product
        softmax = self.softmax(scaled_dot_product)
        dropout = F.dropout(softmax, p=dropout_p)
        output = dropout.matmul(value)
        return output

# Initializing the model
m = Model()

# Inputs to the model
query = torch.randn(3, 4, 5)
key = torch.randn(3, 5, 6)
value = torch.randn(3, 5, 6)
dropout_p = 0.5

