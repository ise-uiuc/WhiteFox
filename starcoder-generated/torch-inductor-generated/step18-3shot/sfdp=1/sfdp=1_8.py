
class Model(torch.nn.Module):
    def __init__(self, d_model):
        super(Model, self).__init__()
        # Attention without masked memory
        self.q = torch.nn.Linear(d_model, d_model, bias=False)
        self.k = torch.nn.Linear(d_model, d_model, bias=False)
        self.v = torch.nn.Linear(d_model, d_model, bias=False)
        self.scale_factor = d_model ** -0.5
        self.atten_dropout = torch.nn.Dropout(0.1)
 
    def forward(self, q, k, v):
        q = self.q(q)
        k = self.k(k)
        v = self.v(v)
        scaled_qk = torch.matmul(q, k.transpose(-2, -1)) * self.scale_factor
        softmax_qk = F.softmax(scaled_qk, dim=-1)
        atten_dropout_qk = self.atten_dropout(softmax_qk)
        output = torch.matmul(atten_dropout_qk, v)
        return output

# Initializing the model
m = Model(512)

# Inputs to the model
q = torch.randn(3, 10, 512)
k = torch.randn(2, 10, 512)
v = torch.randn(2, 10, 512)
res1 = m(q, k, v)

