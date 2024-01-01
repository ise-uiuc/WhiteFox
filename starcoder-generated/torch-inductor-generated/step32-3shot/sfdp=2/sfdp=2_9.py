
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.query_projection = torch.nn.Linear(768, 214)
        self.key_projection = torch.nn.Linear(768, 214)
        self.value_projection = torch.nn.Linear(768, 768)
 
    def forward(self, x1, x2, x3):
        q = self.query_projection(x1)
        k = self.key_projection(x2)
        v = self.value_projection(x3)
        attn = torch.matmul(q, k.transpose(-2, -1))
        inv_scale_factor = torch.FloatTensor(self.key_projection.weight.size()).uniform_() + 1
        scaled_attn = attn.div(inv_scale_factor)
        softmax_attn = scaled_attn.softmax(dim=-1)
        dropout_attn = torch.nn.functional.dropout(softmax_attn, p=dropout_p)
        output = dropout_attn.matmul(v)
        return output

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(8, 8, 768)
x2 = torch.randn(8, 8, 768)
x3 = torch.randn(214, 8, 768)
