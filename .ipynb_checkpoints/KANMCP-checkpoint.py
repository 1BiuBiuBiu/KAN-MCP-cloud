from transformers.models.deberta_v2.modeling_deberta_v2 import DebertaV2PreTrainedModel, DebertaV2Model
import torch
import global_configs
from modules.LinearEncoder import LinearEncoder
from modules.mib import *
from kan import *
from global_configs import DEVICE
from modules.transformer import Transformer
from modules.mlp import MLP
from modules.AutoEncoderCompressor import AutoEncoderCompressor
from modules.LinearEncoder import LinearEncoder


# class KANMCP(DebertaV2PreTrainedModel):
#     def __init__(self, deberta_config, multimodal_config):
#         super().__init__(deberta_config)
#         self.TEXT_DIM = global_configs.TEXT_DIM
#         self.ACOUSTIC_DIM = global_configs.ACOUSTIC_DIM
#         self.VISUAL_DIM = global_configs.VISUAL_DIM
        
#         self.seed = multimodal_config.seed
#         self.compressed_dim = multimodal_config.compressed_dim
#         self.kan_hidden_neurons = multimodal_config.kan_hidden_neurons
#         self.use_KAN_or_MLP = multimodal_config.use_KAN_or_MLP
#         self.use_DRDMIB_or_AE = multimodal_config.use_DRDMIB_or_AE
#         # Deberta
#         Deberta = DebertaV2Model.from_pretrained(multimodal_config.model)
#         self.Deberta = Deberta.to(DEVICE)

#         # DRD-MIB
#         if self.use_DRDMIB_or_AE == 0:      # DRD-MIB
#             self.TEncoder = mib(self.TEXT_DIM, self.compressed_dim, multimodal_config.m_dim)
#             self.AEncoder = mib(self.ACOUSTIC_DIM, self.compressed_dim, multimodal_config.m_dim)
#             self.VEncoder = mib(self.VISUAL_DIM, self.compressed_dim, multimodal_config.m_dim)
#         elif self.use_DRDMIB_or_AE == 1:        # AutoEncoder
#             self.TEncoder = AutoEncoderCompressor(self.TEXT_DIM, multimodal_config.m_dim, self.compressed_dim)
#             self.AEncoder = AutoEncoderCompressor(self.ACOUSTIC_DIM, multimodal_config.m_dim, self.compressed_dim)
#             self.VEncoder = AutoEncoderCompressor(self.VISUAL_DIM, multimodal_config.m_dim, self.compressed_dim)
#         elif self.use_DRDMIB_or_AE == 2:        # Linear Encoder
#             self.TEncoder = LinearEncoder(self.TEXT_DIM, self.compressed_dim, multimodal_config.m_dim)
#             self.AEncoder = LinearEncoder(self.ACOUSTIC_DIM, self.compressed_dim, multimodal_config.m_dim)
#             self.VEncoder = LinearEncoder(self.VISUAL_DIM, self.compressed_dim, multimodal_config.m_dim)
#         elif self.use_DRDMIB_or_AE == 3:        # Transformer Encoder
#             self.TEncoder = Transformer(self.TEXT_DIM, 1, 1, self.compressed_dim, use_cls_token=True)
#             self.AEncoder = Transformer(self.ACOUSTIC_DIM, 1, 1, self.compressed_dim, use_cls_token=True)
#             self.VEncoder = Transformer(self.VISUAL_DIM, 1, 1, self.compressed_dim, use_cls_token=True)

#         if self.use_KAN_or_MLP:
#             self.KAN = KAN(width=[self.compressed_dim * 3, self.kan_hidden_neurons, 1], device="cuda", auto_save=False, seed=self.seed)
#         else:
#             self.MLP = MLP(self.compressed_dim * 3, 1)


#     def forward(
#             self,
#             input_ids,
#             visual,
#             acoustic,
#             label_ids
#     ):
#         # deberta processing text data
#         embedding_output = self.Deberta(input_ids)
#         x_embedding = embedding_output[0]

#         # feature extraction
#         if  self.use_DRDMIB_or_AE == 3:
#             # Transformers expect (seq, batch, dim); grab cls token (latent) as the feature.
#             text_lat, out_t, loss_t = self.TEncoder(x_embedding.transpose(0, 1), label_ids)
#             audio_lat, out_a, loss_a = self.AEncoder(acoustic.transpose(0, 1), label_ids)
#             visual_lat, out_v, loss_v = self.VEncoder(visual.transpose(0, 1), label_ids)

#             text_feature = text_lat[0]
#             audio_feature = audio_lat[0]
#             visual_feature = visual_lat[0]
#         else:
#             x = torch.mean(x_embedding, dim=1)
#             a = torch.mean(acoustic, dim=1)
#             v = torch.mean(visual, dim=1)

#         # feature reduction
#         if self.use_DRDMIB_or_AE == 0:
#             text_feature, out_t, loss_t = self.TEncoder(x, label_ids)
#             audio_feature, out_a, loss_a = self.AEncoder(a, label_ids)
#             visual_feature, out_v, loss_v = self.VEncoder(v, label_ids)
#         elif self.use_DRDMIB_or_AE == 1:
#             text_feature, out_t, loss_t = self.TEncoder(x)
#             audio_feature, out_a, loss_a = self.AEncoder(a)
#             visual_feature, out_v, loss_v = self.VEncoder(v)
#         elif self.use_DRDMIB_or_AE == 2:
#             text_feature, out_t, loss_t = self.TEncoder(x, label_ids)
#             audio_feature, out_a, loss_a = self.AEncoder(a, label_ids)
#             visual_feature, out_v, loss_v = self.VEncoder(v, label_ids)
            
#         # concat
#         concat_feature = torch.cat([text_feature, audio_feature, visual_feature], dim=1)

#         # fusion and predict
#         if self.use_KAN_or_MLP:
#             logits = self.KAN(concat_feature)
#         else:
#             _, _, logits = self.MLP(concat_feature)

#         res = {
#             "logits": logits,
#             "text_feature": text_feature,
#             "audio_feature": audio_feature,
#             "visual_feature": visual_feature,
#             "concat_feature": concat_feature,
#             "loss_t": loss_t,
#             "loss_a": loss_a,
#             "loss_v": loss_v,
#             "out_t": out_t,
#             "out_a": out_a,
#             "out_v": out_v
#         }

#         return res


from transformers.models.deberta_v2.modeling_deberta_v2 import DebertaV2PreTrainedModel, DebertaV2Model
import global_configs
from modules.LinearEncoder import LinearEncoder
from modules.mib import *
from kan import *
from global_configs import DEVICE
from modules.transformer import TransformerEncoder
from transformers import BertModel


class KANMCP(DebertaV2PreTrainedModel):
    def __init__(self, deberta_config, multimodal_config):
        super().__init__(deberta_config)
        self.TEXT_DIM = global_configs.TEXT_DIM
        self.ACOUSTIC_DIM = global_configs.ACOUSTIC_DIM
        self.VISUAL_DIM = global_configs.VISUAL_DIM
        
        self.seed = multimodal_config.seed
        self.compressed_dim = multimodal_config.compressed_dim
        self.kan_hidden_neurons = multimodal_config.kan_hidden_neurons

        self.dataset = multimodal_config.dataset

        self.use_KAN_or_MLP = multimodal_config.use_KAN_or_MLP
        self.use_DRDMIB_or_AE = multimodal_config.use_DRDMIB_or_AE

        # Text encoder: BERT for simsv2, DeBERTa for others
        if getattr(multimodal_config, "dataset", None) == "simsv2":
            self.Deberta = BertModel.from_pretrained(multimodal_config.model).to(DEVICE)
        else:
            Deberta = DebertaV2Model.from_pretrained(multimodal_config.model)
            self.Deberta = Deberta.to(DEVICE)

        # DRD-MIB
        if self.use_DRDMIB_or_AE == 0:      # DRD-MIB
            self.TEncoder = mib(self.TEXT_DIM, self.compressed_dim, multimodal_config.m_dim)
            self.AEncoder = mib(self.ACOUSTIC_DIM, self.compressed_dim, multimodal_config.m_dim)
            self.VEncoder = mib(self.VISUAL_DIM, self.compressed_dim, multimodal_config.m_dim)
        elif self.use_DRDMIB_or_AE == 1:        # AutoEncoder
            self.TEncoder = AutoEncoderCompressor(self.TEXT_DIM, multimodal_config.m_dim, self.compressed_dim)
            self.AEncoder = AutoEncoderCompressor(self.ACOUSTIC_DIM, multimodal_config.m_dim, self.compressed_dim)
            self.VEncoder = AutoEncoderCompressor(self.VISUAL_DIM, multimodal_config.m_dim, self.compressed_dim)
        elif self.use_DRDMIB_or_AE == 2:        # Linear Encoder
            self.TEncoder = LinearEncoder(self.TEXT_DIM, self.compressed_dim, multimodal_config.m_dim)
            self.AEncoder = LinearEncoder(self.ACOUSTIC_DIM, self.compressed_dim, multimodal_config.m_dim)
            self.VEncoder = LinearEncoder(self.VISUAL_DIM, self.compressed_dim, multimodal_config.m_dim)
        elif self.use_DRDMIB_or_AE == 3:        # Transformer Encoder
            self.TEncoder = Transformer(self.TEXT_DIM, 1, 1, self.compressed_dim, use_cls_token=True)
            self.AEncoder = Transformer(self.ACOUSTIC_DIM, 1, 1, self.compressed_dim, use_cls_token=True)
            self.VEncoder = Transformer(self.VISUAL_DIM, 1, 1, self.compressed_dim, use_cls_token=True)

        self.KAN = KAN(width=[self.compressed_dim * 3, self.kan_hidden_neurons, 1], device="cuda", auto_save=False, seed=self.seed)

    def forward(
            self,
            input_ids,
            visual,
            acoustic,
            label_ids
    ):
        # deberta processing text data
        if self.dataset != "simsv2":
            embedding_output = self.Deberta(input_ids)
            x_embedding = embedding_output[0]
        else:
            # simsv2 already stores text features; skip DeBERTa and use them directly
            x_embedding = input_ids.float()

        # feature extraction
        if  self.use_DRDMIB_or_AE == 3:
            # Transformers expect (seq, batch, dim); grab cls token (latent) as the feature.
            text_lat, out_t, loss_t = self.TEncoder(x_embedding.transpose(0, 1), label_ids)
            audio_lat, out_a, loss_a = self.AEncoder(acoustic.transpose(0, 1), label_ids)
            visual_lat, out_v, loss_v = self.VEncoder(visual.transpose(0, 1), label_ids)

            text_feature = text_lat[0]
            audio_feature = audio_lat[0]
            visual_feature = visual_lat[0]
        else:
            if self.dataset == "simsv2":
                # For simsv2 the text features are already pooled; keep feature dim intact
                x = x_embedding
            else:
                x = torch.mean(x_embedding, dim=1)
            a = torch.mean(acoustic, dim=1)
            v = torch.mean(visual, dim=1)

        # feature reduction
        if self.use_DRDMIB_or_AE == 0:
            text_feature, out_t, loss_t = self.TEncoder(x, label_ids)
            audio_feature, out_a, loss_a = self.AEncoder(a, label_ids)
            visual_feature, out_v, loss_v = self.VEncoder(v, label_ids)
        elif self.use_DRDMIB_or_AE == 1:
            text_feature, out_t, loss_t = self.TEncoder(x)
            audio_feature, out_a, loss_a = self.AEncoder(a)
            visual_feature, out_v, loss_v = self.VEncoder(v)
        elif self.use_DRDMIB_or_AE == 2:
            text_feature, out_t, loss_t = self.TEncoder(x, label_ids)
            audio_feature, out_a, loss_a = self.AEncoder(a, label_ids)
            visual_feature, out_v, loss_v = self.VEncoder(v, label_ids)
        # concat
        concat_feature = torch.cat([text_feature, audio_feature, visual_feature], dim=1)

        # fusion and predict
        logits = self.KAN(concat_feature)

        res = {
            "logits": logits,
            "text_feature": text_feature,
            "audio_feature": audio_feature,
            "visual_feature": visual_feature,
            "concat_feature": concat_feature,
            "loss_t": loss_t,
            "loss_a": loss_a,
            "loss_v": loss_v,
            "out_t": out_t,
            "out_a": out_a,
            "out_v": out_v
        }

        return res