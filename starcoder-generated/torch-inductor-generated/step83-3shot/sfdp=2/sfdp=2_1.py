
class Model(torch.nn.Module):
    def __init__(self, input_dim, output_dim, dropout_p, inv_scale_factor, num_heads):
        super().__init__()
        self.query = torch.nn.Linear(input_dim, output_dim) # Define a feed-forward network that maps keys and queries together
        self.key = torch.nn.Linear(input_dim, output_dim)
        self.value = torch.nn.Linear(input_dim, output_dim)
 
    def forward(self, x1, x2):
        query = self.query(x1)
        key = self.key(x2)
        value = self.value(x2)
 
        qk = torch.matmul(query, key.transpose(-2, -1))
        scaled_qk = qk.div(inv_scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=dropout_p)
        output = dropout_qk.matmul(value)
 
        return output

# Initializing the model
m = Model(3, 4, 0.5, 1.0, 2)

# Inputs to the model
x1 = torch.randn(1, 3)
x2 = torch.randn(1, 2, 5)
