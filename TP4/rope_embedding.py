import torch

class RoPE:
    @staticmethod
    def get_cos_sin_(
        embedding_size: int,
        precomp_len: int = 4096,
        theta_param: float = 10000,
        factor: float = 1
    ):
        theta = factor / (
              theta_param ** (torch.arange(0, embedding_size, 2).float() / embedding_size)
        )
        seq_idx = torch.arange(
            0, precomp_len, 1.0
        )   
        rotation_matrix = seq_idx.unsqueeze(1)@theta.unsqueeze(0)

        return (
            rotation_matrix.cos().unsqueeze(-1).repeat(1,1 ,2).view(precomp_len, -1,1).squeeze(),
            (rotation_matrix.sin().unsqueeze(-1).repeat(1,1 ,2) * torch.Tensor([-1,1])).view(precomp_len, -1, 1).squeeze()
        )

    def __init__(
        self,
        embedding_size: int,
        precomp_len: int = 4096,
        theta_param: float = 10000,
        factor: float = 1
    ):
        assert embedding_size%2 == 0, "RoPE is only implemented for dimension being multiple of 2"
    
        self.embedding_size = embedding_size
        self.theta_param = theta_param 
        self.factor = factor
        
        self.rope_cos, self.rope_sin =\
            RoPE.get_cos_sin_(
                self.embedding_size,
                precomp_len,
                self.theta_param,
                self.factor
            )

    def __call__(self, x: torch.Tensor, start_pos_id: int  = 0):
        '''
            Apply RoPE embedding to a tensor x
            
            Parameters
            x : [torch.Tensor]
                Expected dim BxHxLxN
            start_pos_id : int
                The start of the sequence to apply RoPE
        '''
        B, H, L, N = x.shape # batch size, sequence len, embed dim
        if L > self.rope_cos.shape[0]:
            # The precomputed RoPE has a too small size
            self.rope_cos, self.rope_sin =\
                RoPE.get_cos_sin_(
                    self.embedding_size,
                    L,
                    self.theta_param,
                    self.factor
                )
        if (x.device != self.rope_cos.device):
            # if precomputed tensor and rope are not on the same device
            self.rope_cos = self.rope_cos.to(x.device)
            self.rope_sin = self.rope_sin.to(x.device)

        return ( x * self.rope_cos[start_pos_id: start_pos_id + L] 
                 +
                 x.reshape(B, H, L, N//2, 2).flip(-1).view(B, H, L, N) * self.rope_sin[start_pos_id: start_pos_id + L] 
               )