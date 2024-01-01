
def get_model(in_channels, out_channels):
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.softmax = torch.nn.Softmax(dim=-1)
            self.dropout = torch.nn.Dropout(p=0.3)
        
        def forward(self, x3):
            v1 = torch.matmul(query, key.transpose(-2, -1))
            v2 = v1.div(inv_scale_factor)
            v3 = self.softmax(v2)
            v4 = self.dropout(v3)
            return v4
 
    return Model()

# Initializing the model
m = get_model(128, 256)

# Inputs to the model
query = torch.randn(1, 4, 128)
key = torch.randn(1, 3, 128)
inv_scale_factor = 1 / math.sqrt(key.size(-1))
value = torch.randn(1, 3, 256)
dropout_p = 0.3

