
class Model(torch.nn.Module):
    def __init__(self, hparams: argparse.Namespace):
        super().__init__()
        self.d_q = hparams.d_q
        self.d_k = hparams.d_k
        self.softmax_temp = hparams.softmax_temp
        self.d_v = hparams.d_v
        self.d_model = hparams.d_model
        self.dropout = hparams.dropout
        self.scale_factor = math.sqrt(self.d_k)
        self.q_linear = torch.nn.Linear(hparams.d_input, self.d_q)
        self.k_linear = torch.nn.Linear(hparams.d_input, self.d_k)
        self.v_linear = torch.nn.Linear(hparams.d_input, self.d_v)
        self.projection = torch.nn.Linear(self.d_v, self.d_model)
 
    def forward(self, x):
        q, k, v = self.q_linear(x), self.k_linear(x), self.v_linear(x)
        scores = torch.matmul(q, k.transpose(-2, -1)).mul(self.scale_factor).softmax(dim=-1)
        p_attn = F.dropout(scores, p=self.dropout, training=self.training)
        context = torch.matmul(p_attn, v)
        output = self.projection(context)
        return output, p_attn

# Initializing the default parameters
hparams = argparse.Namespace()
hparams.d_q = 10
hparams.d_k = 20
hparams.softmax_temp = 0.2
hparams.d_v = 30
hparams.d_input = 40
hparams.dropout = 0.1
hparams.d_model = 80

# Initializing the model
m = Model(hparams)

# Inputs to the model
x = torch.randn(100, 40)
__output__,__scores__ = m(x)

