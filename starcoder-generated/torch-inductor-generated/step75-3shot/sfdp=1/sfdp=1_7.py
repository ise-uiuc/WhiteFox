
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.softmax1 = torch.nn.Softmax(dim=-1)
        self.dropout1 = torch.nn.Dropout(0.4418341416309934)
        self.softmax2 = torch.nn.Softmax(dim=-1)
        self.dropout2 = torch.nn.Dropout(0.10262331089621027)
        self.softmax3 = torch.nn.Softmax(dim=-1)
        self.dropout3 = torch.nn.Dropout(0.5589416762784175)
        self.softmax4 = torch.nn.Softmax(dim=-1)
        self.dropout4 = torch.nn.Dropout(0.1053028797429638)

    def forward(self, query, key, value, scale_factor, inv_scale_factor, dropout_p):
        qk = torch.matmul(query, key.transpose(-2, -1))
        v1 = self.softmax1(qk.div(inv_scale_factor))
        v2 = self.dropout1(v1)
        v3 = torch.matmul(v2, value)
        v4 = self.softmax2(qk.div(inv_scale_factor))
        v5 = self.dropout2(v4)
        v6 = torch.matmul(v5, value)
        v7 = self.softmax3(qk.div(inv_scale_factor))
        v8 = self.dropout3(v7)
        v9 = torch.matmul(v8, value)
        v10 = self.softmax4(qk.div(inv_scale_factor))
        v11 = self.dropout4(v10)
        v12 = torch.matmul(v11, value)
        return v3, v6, v9, v12

# Initializing the model
m = Model()

# Inputs to the model
query = torch.randn(16, 4, 10, 64)
key = torch.randn(16, 4, 7, 64)
value = torch.randn(16, 4, 7, 64)
scale_factor = torch.randn(16, 4, 7, 7)
inv_scale_factor = torch.randn(16, 4, 7, 7)
dropout_p = torch.tensor(0.8)
__output1__, __output2__, __output3__, __output4__ = m(query, key, value, scale_factor, inv_scale_factor, dropout_p)

