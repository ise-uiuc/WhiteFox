
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dropout = torch.nn.Dropout(0.5)
    
    def forward(self, in_features_1):
        v1 = torch.matmul(query, key.transpose(-2, -1))
        v2 = v1.div(inv_scale_factor)
        v3 = v2.softmax(dim=-1)
        v4 = self.dropout(v3)
        output = v4.matmul(value)
        return output

# Initializing the model
m = Model()

# Inputs to the model
query = torch.randn(1, 3, 64, 1)
key = torch.randn(1, 3, 1, 64)
inv_scale_factor = torch.randn(1)
value = torch.randn(1, 3, 1, 64)
dropout_p = 0.5
