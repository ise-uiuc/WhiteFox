
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, query, key, value, inv_scale_factor, dropout_p):
        qk = torch.matmul(query, key.transpose(-2, -1))
        scaled_qk = qk.div(inv_scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=dropout_p)
        output = dropout_qk.matmul(value)
        return output
    
def _make_pipelined_pipelined_bert_encoder_layer(self, config, drop_out_rate):
    return nn.ModuleList([_PipelinedPipelinedBertEncoderLayer(config, drop_out_rate=drop_out_rate)
                           for _ in range(config.num_hidden_layers)])

class _PipelinedPipelinedBertEncoderLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.attention = _PipelinedPipelinedBertAttention(config)
        self.output = _BertOutput(config)
        if self.is_decoder:
            self.crossattention = _PipelinedPipelinedBertAttention(config)
            self.crossoutput = _BertOutput(config)

# Initializing the model
m = _PipelinedPipelinedBertEncoderLayer()

# Inputs to the model
query = torch.randn(20, 8, 32)
key = torch.randn(20, 8, 32)
value = torch.randn(20, 8, 32)
inv_scale_factor = torch.randn(20, 8, 1)
dropout_p = 0.1
