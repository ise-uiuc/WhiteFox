
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(0)
        self.query = torch.randn(4, 20, 50)
        self.key = torch.randn(4, 40, 50)
        self.value = [self.value for i in range(3)]
        self.scale_factor = torch.randn(1)
        self.dropout_p = torch.randn(1)
 
    def forward(self, query, key, value, scale_factor, dropout_p):
        qk = torch.matmul(query, key.transpose(-2, -1))
        scaled_qk = qk.mul(scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=dropout_p)
        output = dropout_qk.matmul(value)
        return output
 
m = Model()
