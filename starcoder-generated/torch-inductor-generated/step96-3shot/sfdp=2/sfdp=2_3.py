
class Model(torch.nn.Module):
    def __init__(self, dropout_p=0.5, attention_dropout_p=0.3):
        super().__init__()
        self.key = torch.autograd.Variable(torch.randn(8, 6, 3), requires_grad=False)
        self.inv_scale_factor = torch.sqrt(torch.Tensor([8.0]))
        self.dropout_p = dropout_p
        self.attention_dropout_p = attention_dropout_p
 
    def forward(self, x1):
        v1 = torch.matmul(x1, self.key.transpose(-2, -1))
        v2 = v1.div(self.inv_scale_factor)
        v3 = v2.softmax(dim=-1)
        v4 = torch.nn.functional.dropout(v3, p=self.dropout_p)
        v5 = torch.matmul(v4, value)
        return v5
 
# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 8, 10)
