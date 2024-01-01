
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dropout_11 = torch.nn.Dropout(0.1)
        self.softmax_6 = torch.nn.Softmax(dim=-1)
        self.dropout_26 = torch.nn.Dropout(0.1)
        self.softmax_7 = torch.nn.Softmax(dim=-1)
        self.matmul_8 = torch.matmul()
        self.dropout_9 = torch.nn.Dropout(0.1)
        self.matmul_10 = torch.matmul()
 
    def forward(self, v4, v5, v6):
        v1 = self.dropout_11(v4)
        v2 = self.softmax_6(v1)
        v3 = v2 / 0.05241306459982384
        v11 = self.dropout_26(v5)
        v12 = self.softmax_7(v11)
        v13 = v12 / 0.00521668520555902
        v7 = self.matmul_8(v3, v6)
        v8 = self.dropout_9(v7)
        v9 = self.matmul_10(v13, v8)
        return v9

# Initializing the model
m = Model()

# Inputs to the model
v4 = torch.randn(1, 1024, 64)
v5 = torch.randn(1, 5, 512)
v6 = torch.randn(5, 512, 512)
