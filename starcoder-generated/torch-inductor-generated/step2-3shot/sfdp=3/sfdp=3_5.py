
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.attention = torch.nn.multihead_attention_forward

    def forward(
            self,
            m1,
            m2,
            m3):
        qk = torch.matmul(m1, m2.transpose(-2, -1))
        scaled_qk = qk.mul(0.125)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=0.1)
        output = self.attention(
            query=dropout_qk,
            key=dropout_qk,
            value=dropout_qk,
            key_padding_mask=None,
            need_weights=False,
            attn_mask=None,
            use_separate_proj_weight=False,
            q_proj_weight=m3,
            k_proj_weight=m3,
            v_proj_weight=None,
            static_k=None,
            static_v=None,
            attn_output_weights=None,
            out=None
        )
        return output.unsqueeze(0)
        
# Initializing the model
m = Model()

