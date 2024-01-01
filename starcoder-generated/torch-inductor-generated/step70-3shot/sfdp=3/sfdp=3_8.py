
class Model(torch.nn.Module):
    def __init__(self, embedding_dim=16, heads=2, dropout_p=0.0, scale_factor=1.0 / 8.0):
        super().__init__()
        self.softmax = torch.nn.Softmax(dim=-1)
        self.matmul = torch.nn.Linear(embedding_dim * heads, 1, bias=False)
        self.matmul.weight.requires_grad = False
        self.matmul.weight.set_(torch.zeros_like(self.matmul.weight))
        self.matmul.weight[0, 0] = -1 * scale_factor
 
    def forward(self, query, key, value, mask=None):
        qk = torch.matmul(query, key.transpose(-2, -1))
        qk_summed = qk.sum(dim=-1, keepdim=True)
        qk_summed = qk_summed + qk_summed.transpose(-2, -1)
 
        if mask is not None:
            mask = mask.view(qk_summed.shape)
            qk_summed.masked_fill(mask, float("-inf"))
 
        softmax_qk = self.softmax(qk_summed)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=dropout_p)
        output = dropout_qk.matmul(value)
 
        return output

# Initializing the model
import numpy as np
m = Model()

# Inputs to the model

q = torch.randn(1, 1, 8)
k = torch.randn(1, 1, 8)
v = torch.randn(1, 1, 8)
mask = torch.tensor(np.random.randint(0, 2, (1, 1, 8)), dtype=torch.bool)
