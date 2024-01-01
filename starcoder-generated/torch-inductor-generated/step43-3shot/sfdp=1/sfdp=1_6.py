
class Model(torch.nn.Module):
    def __init__(self, query, key, value, dropout_p, inv_scale_factor):
        super().__init__()
        self.mat = torch.nn.functional.linear
        self.qk = self.mat(query, key.transpose(-2, -1))
        self.scaled_qk = self.qk.div(inv_scale_factor)
        self.softmax_qk = torch.nn.functional.softmax(self.scaled_qk, dim=-1)
        self.dropout_qk = torch.nn.functional.dropout(self.softmax_qk, p=dropout_p)
        self.output = self.dropout_qk.matmul(value)
 
    def forward(self, query, key, value, dropout_p, inv_scale_factor):
        output = self.mat(query, key.transpose(-2, -1))
        output = output.div(inv_scale_factor)
        output = torch.nn.functional.softmax(output, dim=-1)
        output = torch.nn.functional.dropout(output, p=dropout_p)
        output = self.mat(query, value)
        return output
 
# Initializing the model
m = Model(query, key, value, dropout_p, inv_scale_factor)
 
# Inputs to the model
x1 = x2 = x3 = torch.randn(1, 1152, 4, 64)
