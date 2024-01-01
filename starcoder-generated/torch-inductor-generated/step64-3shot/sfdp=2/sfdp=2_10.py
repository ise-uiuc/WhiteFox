
class MyAttention(nn.Module):

    def forward(self, q, k, v):
        scores = torch.matmul(q, k.transpose(-2, -1))
        inv_scale_factor = 1. / math.sqrt(scores.size(-1))
        scaled_scores = scores * inv_scale_factor
        softmaxed_probs = scaled_scores.softmax(dim=-1)
        d_model = q.size(-1)
        dropout_probs = torch.nn.functional.dropout(softmaxed_probs, p=dropout_p, training=self.training)
        output = torch.matmul(dropout_probs, v)

        return output, dropout_probs

class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        dim = 256
        self.query_proj = torch.nn.Linear(dim, dim)
        self.key_proj = torch.nn.Linear(dim, dim)
        self.value_proj = torch.nn.Linear(dim, dim)
        self.dot_attention = MyAttention()
 
    def forward(self, q, k, v):
        q = self.query_proj(q)
        k = self.key_proj(k)
        v = self.value_proj(v)
        output, dropout_probs = self.dot_attention(q, k, v)
        return output, dropout_probs

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
x2 = torch.randn(1, 3, 64, 64)
x3 = torch.randn(1, 3, 64, 64)
__output__, __probs__ = m(x1, x2, x3)


# Inputs to the model
x1 = torch.randn(1, 3, 256)
x2 = torch.randn(1, 3, 256)
__output__, __probs__ = m(x1, x2)

def test():
    print('hello')
