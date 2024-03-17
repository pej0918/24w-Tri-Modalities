import math 
import torch
import torch.nn as nn
import torch.nn.functional as F 

class ScaleDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaleDotProductAttention, self).__init__()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, q, k, v, e=1e-12):
        # input size: [batch_size, head, length, d_tensor]
        batch_size, head, length, d_tensor = k.size()

        k_t = k.transpose(2,3)
        score = (q @ k_t) / math.sqrt(d_tensor)  # scaled dot product 
        score = self.softmax(score)  #[0,1]
        v = score @ v

        return v, score 
    
class MultiHeadCrossAttention(nn.Module):
    def __init__(self, d_model, n_head):
        super(MultiHeadCrossAttention, self).__init__()
        self.d_model = d_model
        self.n_head = n_head
        self.attention = ScaleDotProductAttention()

        self.w_k = nn.Linear(d_model, d_model)
        self.w_q = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_concat = nn.Linear(d_model, d_model)
    
    def forward(self, k, q, v):
        q, k, v = self.w_q(q), self.w_k(k), self.w_v(v)
        q, k, v = self.split(q), self.split(k), self.split(v)
        out, attention = self.attention(q, k, v)
        out = self.concat(out)
        out = self.w_concat(out)
        return out 

    def split(self, tensor):
        # [batch_size, length, d_model] -> [batch_size, head, length, d_model]
        batch_size, length, d_model = tensor.size()
        d_tensor = d_model // self.n_head
        tensor = tensor.view(batch_size, length, self.n_head, d_tensor).transpose(1,2)
        return tensor 
    
    def concat(self, tensor):
        batch_size, head, length, d_tensor = tensor.size()
        tensor = tensor.transpose(1,2).contiguous().view(batch_size, length, self.d_model)
        return tensor

if __name__ == '__main__':
    d_model = 2048
    n_head = 8
    multi_head_cross_attention = MultiHeadCrossAttention(d_model, n_head)

    # video & text fature 
    v_feature = torch.rand(1, 1, d_model)
    t_feature = torch.rand(1, 1, d_model)

    cross_attention_v_t = multi_head_cross_attention(v_feature,t_feature,v_feature)
    cross_attention_t_v = multi_head_cross_attention(t_feature,v_feature,t_feature)

    output = torch.cat((cross_attention_v_t, cross_attention_t_v), dim=2)