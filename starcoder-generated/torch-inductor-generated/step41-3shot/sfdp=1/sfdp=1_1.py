
class Model(torch.nn.Module):
    def __init__(self, d_model, num_heads, dropout_p):
        super().__init__()

        self.qk_w = torch.nn.Linear(d_model, d_model)
        self.qk_b = torch.nn.Parameter(torch.zeros((d_model)), requires_grad=True)
        self.v_w = torch.nn.Linear(d_model, d_model)
        self.v_b = torch.nn.Parameter(torch.zeros((d_model)), requires_grad=True)

        self.dropout_p = dropout_p

        self.softmax = torch.nn.Softmax(dim=-1)

    @property
    def num_heads(self):
        return len(self.qk_w) // 3

    def forward(self, query, value):
        qk = torch.matmul(self.qk_w(query) + self.qk_b, self.v_w(value).transpose(-2, -1))

        inv_scale_factor = math.sqrt(query.shape[-1] / self.num_heads)
        scaled_qk = qk.div(inv_scale_factor)

        softmax_qk = self.softmax(scaled_qk)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=self.dropout_p)

        output = dropout_qk @ self.v_w(value)

        return output

# Initializing the model
m = Model(16, 2, 0.1)

# Inputs to the model
query = torch.randn(16, 16, 4)
value = torch.randn(16, 16, 4)
