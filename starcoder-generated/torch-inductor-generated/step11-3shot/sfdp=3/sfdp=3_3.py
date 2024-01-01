
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.softmax = torch.nn.Softmax()
 
 
    def forward(self, query, key, scale_factor, dropout_p, value):
        # Computing the dot product
        v1 = torch.matmul(query, key.transpose(-2, -1))
        # Scaling the dot product
        v2 = v1 * scale_factor
        # Softmax
        v3 = self.softmax(v2, dim=-1)
        # Dropout
        v4 = torch.dropout(v3, p=dropout_p)
        # Matrix multiplication
        output = torch.matmul(v4, value)
        return v4

# Initializing the model
m = Model()

# Input to the model
query = key = scale_factor = dropout_p = value = torch.randn(6, 5, 7)
