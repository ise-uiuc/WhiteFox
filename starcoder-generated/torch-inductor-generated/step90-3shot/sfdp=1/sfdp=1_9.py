
class Model(torch.nn.Module):
    def __init__(self, dropout_p=.1):
        super().__init__()
        self.dropout_p = dropout_p
 
    def forward(self, y1, y2):
        q = y1.unsqueeze(2).mul(y2.unsqueeze(1)).sum(-1)
        inv_scale_factor = y2.size(-1) ** -0.25
        q = q * inv_scale_factor
        q = q.softmax(dim=-1)
        q = torch.nn.functional.dropout(q, p=self.dropout_p)
        output = q.mul(y2).sum(-2).unsqueeze(-2)
        return output

# Initializing the model
m = Model()

# Inputs to the model
y1 = torch.randn(4, 13)
y2 = torch.randn(4, 13, 15)
