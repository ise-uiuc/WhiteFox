
class Model(torch.nn.Module):
    def __init__(self, num_queries=5, hidden_dim=16, value_dim=16, inv_scale_factor=512, dropout_p=0.1):
        super(Model, self).__init__()
        self.dropout_p = dropout_p
        self.inv_scale_factor = inv_scale_factor

        self.query = torch.nn.Linear(num_queries, hidden_dim)
        self.key = torch.nn.Linear(num_queries, hidden_dim)
        self.value = torch.nn.Linear(num_queries, value_dim)

    def forward(self, x1):
        qk = self.query(x1).unsqueeze(-2) * self.key(x1).unsqueeze(-2).transpose(-2, -1)
        scaled_qk = qk.div(self.inv_scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        return torch.nn.functional.dropout(softmax_qk, p=self.dropout_p)

# Initializing the model
m = Model(num_queries=5)

# Inputs to the model
x1 = torch.randn(1, 5)
