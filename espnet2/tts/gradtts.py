import torch
from espnet2.tts.abs_tts import AbsTTS
from espnet.nets.pytorch_backend.transformer.encoder_layer import EncoderLayer

class Encoder(torch.nn.Module):
    def __init__(
        self,
        idim,
        adim: int = 384,
        aheads: int = 4,
        elayers: int = 6,
        eunits: int = 1536,
    ):
        super().__init__()
        # self.encoders = repeat(
        #     num_blocks,
        #     lambda lnum: EncoderLayer(
        #         attention_dim,
        #         encoder_selfattn_layer(*encoder_selfattn_layer_args[lnum]),
        #         positionwise_layer(*positionwise_layer_args),
        #         dropout_rate,
        #         normalize_before,
        #         concat_after,
        #     ),
        # )



class GradTTS(AbsTTS):
    def __init__(self):
        super(GradTTS, self).__init__()