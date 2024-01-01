
class Model(torch.nn.Module):
    def __init__(self, inv_scale_factor, dropout_p):
        super().__init__()
        self.query = torch.nn.Linear(3, 4)
        self.key = torch.nn.Linear(3, 5)
        self.value = torch.nn.Linear(5, 7)
        self.inv_scale_factor = inv_scale_factor
        self.dropout_p = dropout_p
 
    def forward(self, x):
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)
        QK = (Q @ K.transpose(-2, -1)) / self.inv_scale_factor
        attention_map = QK.softmax(dim=-1)
        attention_mask = F.dropout(attention_map, p=self.dropout_p)
        self.output(attention_mask @ V)
        return self.output
 
# Initializing the model
m = Model(0.2, 0.3)

# Inputs to the model
x = torch.randn(2, 3)
