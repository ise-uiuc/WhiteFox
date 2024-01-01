
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        from torch.nn.init import constant_
        self.q = torch.nn.Parameter(torch.randn(3, 512) * 0.01)
        self.k = torch.nn.Parameter(torch.randn(512, 1024) * 0.01)
        self.v = torch.nn.Parameter(torch.randn(3, 512) * 0.01)
        const = torch.randn(512) * 0.01
        # Here we assign the values of PyTorch modules to Torch parameters, and we can assign different weights with the 'torch.nn.Parameter' function.
        weights = [torch.nn.Parameter(w, True) for w in [const] * 8]
        self.attn_dropout = torch.nn.AlphaDropout(p=0.5)
        self.softmax_dropout = torch.nn.AlphaDropout(p=0.5)
        self.fc = torch.nn.Linear(2048, 1000, bias=False)
        self.weight_softmax = torch.nn.Parameter(torch.randn(256, 256), requires_grad=False)
        self.weight_softmax = torch.nn.Parameter(torch.Tensor(weights), requires_grad=False)
        self.scale_factor = 1.0 / self.weight_softmax.size(1)**0.5
 
    def forward(self, q, k):
        m1 = torch.matmul(q, self.k)
        m2 = m1.div(self.scale_factor)
        m3 = m2.softmax(dim=-1)
        m4 = self.softmax_dropout(m3)
        m5 = torch.matmul(m4, self.v)
        m6 = self.attn_dropout(m5)
        m7 = m6.unsqueeze(1)
        m8 = self.weight_softmax.transpose(0, 1)
        m9 = m7.matmul(m8)
        m10 = m9.squeeze(1)
        m11 = self.fc(m10)
        return m11

# Initializing the model
m = Model()

# Inputs to the model
q = torch.randn(5, 3, 512)
k = torch.randn(6, 512, 1024)
