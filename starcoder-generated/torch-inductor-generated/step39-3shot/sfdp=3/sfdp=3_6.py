
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.scale_factor = nn.Parameter(torch.ones(1, 1, 1))
 
    def forward(self, attention_mask, att_output):
        attention_mask = attention_mask.softmax(dim=-1)
        attention_mask = torch.nn.functional.dropout(attention_mask, p=dropout_p)
        output = attention_mask.matmul(att_output)
        return output

# Initializing the model
m = Model()

# Inputs to the model
attention_mask = torch.tensor(np.ones((1, 1, 100, 100), np.float32))
att_output = torch.randn(1, 1, 100, 5)
output = m(attention_mask, att_output)

