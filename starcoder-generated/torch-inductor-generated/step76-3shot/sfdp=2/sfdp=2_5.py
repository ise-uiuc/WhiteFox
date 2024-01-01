
class Model(torch.nn.Module):

    def __init__(self):
        super(Model, self).__init__()
        self.p_dropout = 0.6
        self.q_dropout = 0.4
        self.k_dropout = 0.2
        self.d_v = 32

        self.q_linear = torch.nn.Linear(32, self.d_v)
        self.k_linear = torch.nn.Linear(16, self.d_v)
        self.v_linear = torch.nn.Linear(16, self.d_v)
        self.o_linear = torch.nn.Linear(self.d_v, 8)

    def forward(self, q, k, v):
        inv_scale = math.sqrt(self.d_v)
        q = torch.nn.functional.dropout(self.q_linear(q), p=self.q_dropout)
        k = torch.nn.functional.dropout(self.k_linear(k), p=self.k_dropout)
        v = torch.nn.functional.dropout(self.v_linear(v), p=self.k_dropout)
        q = q / inv_scale
        scores = torch.bmm(q, k.transpose(1, 2))
        attentions = torch.nn.functional.dropout(torch.nn.Softmax(dim=-1)(scores), p=self.p_dropout)
        out_linear = torch.bmm(attentions, v)
        out = torch.nn.functional.dropout(self.o_linear(out_linear), p=self.p_dropout)
        return out

# Initializing the model
m = Model()

# Inputs to the model
q = torch.randn(10, 16)
k = torch.randn(10, 16)
v = torch.randn(10, 16)
