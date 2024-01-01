
class Model(torch.nn.Module):
    def __init__(self, D_key, D_val, N, D_model, H, D_inner):
        super().__init__()
        self.linear = torch.nn.Linear(D_model, H)
        self.linear.weight.data.normal_(mean=0, std=1)
        self.linear.bias.data.zero_()
        self.scale_factor = np.sqrt(D_key)
        self.dropout = torch.nn.Dropout(p=dropout_p)
 
    def forward(self, x1, x2, x3):
        x1 = self.linear(x1)
        x2 = x2 * self.scale_factor
        x3 = x3.transpose(-2, -1)
        v1 = torch.matmul(x1, x2)
        v2 = v1.mul(self.scale_factor).softmax(dim=-1)
        v3 = self.dropout(v2)
        out = torch.matmul(x3, v3)
        return out

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randint(0, 10, (M, D_model))
x2 = f(x1)
x3 = torch.randn(N, D_model)
