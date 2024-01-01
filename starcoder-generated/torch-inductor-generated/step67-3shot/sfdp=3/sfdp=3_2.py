
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.mul = __torch__.atg.torch.mul
        self.dropout = __torch__.atg.torch.nn.functional.dropout
        self.matmul = __torch__.atg.torch.matmul
        self.softmax = __torch__.atg.torch.softmax
        self.addmm = __torch__.atg.torch.addmm
        self.transpose = __torch__.atg.torch.transpose
        self.matmul2 = __torch__.atg.torch.matmul
        self.addmm2 = __torch__.atg.torch.addmm

    def forward(self, query, key, value, scale_factor, dropout_p):
        qk = self.matmul(query, self.transpose(key, -2, -1))
        scaled_qk = qk * scale_factor
        softmax_qk = self.softmax(scaled_qk, -1)
        dropout_qk = self.dropout(softmax_qk, p=dropout_p)
        output = self.addmm(dropout_qk, value)
        output2 = self.addmm2(self.transpose(dropout_qk, -2, -1), query)
        return output + output2

# Initializing the model
m = Model()

# Inputs to the model
query = torch.randn(1, 64, 512)
key = torch.randn(1, 64, 512)
value = torch.randn(1, 64, 512)
scale_factor = torch.randn(1, 1, 512)
dropout_p = 0.1
