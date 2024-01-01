
class Model(torch.nn.Module):
    def __init__(self, model_config):
        super().__init__()
        self.model_config = model_config
 
    def forward(self, q, k, v, m, m_mask):
        qk = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(q.size(-1))
        scaled_qk = qk.div(self.model_config.scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=self.model_config.dropout_p)
        output = dropout_qk.matmul(v)
        return output

# Initializing the model
m_config = ModelConfig.from_dict(json.load(open('config.json', 'r')))
m = Model(model_config=m_config)

# Inputs to the model
q = torch.randn(1, 64, 64)
k = torch.randn(1, 64, 64)
v = torch.randn(1, 64, 64)
m = m_config.mem_dim
m_mask = torch.ones(1, 64, 6, device='cpu')
