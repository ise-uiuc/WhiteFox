
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.in_features = 10
        self.out_features = 20
        self.num_heads = 3
    
    def forward(self, x1, x2, x3):
        q = torch.randn(1, self.num_heads * self.in_features, 256)
        k = torch.randn(1, self.num_heads * self.in_features, 256)
        v = torch.randn(1, self.num_heads * self.in_features, 256)
        inv_scale = 1. / np.sqrt(self.in_features * self.out_features)
        qk = torch.matmul(q, k.transpose(-2, -1))
        scaled_qk = qk.div(inv_scale)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=0.5)
        output = dropout_qk.matmul(v)
        return output

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 3, 256)
x2 = torch.randn(1, 3, 256)
x3 = torch.randn(1, 3, 256)
