
from torch.nn import Dropout, Softmax
class Model(torch.nn.Module):
    def __init__(self, n_head):
        super().__init__()
        self.w_q = torch.nn.Linear(768, 768)
        self.w_k = torch.nn.Linear(768, 768)
        self.w_v = torch.nn.Linear(768, 768)
        self.attention_dropout = Dropout(attention_dropout_p)
        self.attention_softmax = Softmax(dim=-1)
        self.v = torch.nn.Linear(768, 768)
        self.dropout = Dropout(dropout_p)
 
    def forward(self, q, k, v):
        q1 = self.w_q(q)
        k1 = self.w_k(k)
        v1 = self.w_v(v)
        q2 = q.view(q.size(0), q.size(1), self.n_head, -1).transpose(1, 2)
        k2 = k.view(k.size(0), k.size(1), self.n_head, -1).transpose(1, 2)
        v2 = v.view(v.size(0), v.size(1), self.n_head, -1).transpose(1, 2)
        q3 = q1.view(q.size(0), q.size(1), self.n_head, -1).transpose(1, 2)
        k3 = k1.view(k.size(0), k.size(1), self.n_head, -1).transpose(1, 2)
        v3 = v1.view(v.size(0), v.size(1), self.n_head, -1).transpose(1, 2)
        q4 = torch.matmul(q2, k3.transpose(-1, -2)) 
        q5 = q4 * scale_factor # scale_factor is a tensor with shape (1,1,768,768)
        q6 = self.attention_softmax(q5)
        q7 = self.attention_dropout(q6)
        output = torch.matmul(q7, v3)
        output = output.transpose(1, 2).contiguous().view(output.size(0), -1, 768)
        output = self.v(output)
        output = self.dropout(output)
        return output

# Initializing the model
model = Model(n_head)

# Inputs to the model
q = torch.randn(1, 18, 768)
k = torch.randn(1, 20, 768)
v = torch.randn(1, 20, 768)
