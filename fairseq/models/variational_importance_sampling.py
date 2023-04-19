# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from fairseq import checkpoint_utils
from fairseq.data.legacy.masked_lm_dictionary import MaskedLMDictionary
from fairseq.models import register_model, register_model_architecture
from fairseq.models.transformer import (
    TransformerDecoder,
    TransformerEncoder,
    TransformerModel,
    base_architecture as transformer_base_architecture,
)

@register_model("variational_is")
class Variational_Importance_Sampling(TransformerModel):
    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        TransformerModel.add_args(parser)
        parser.add_argument(
            "--pretrained-esm-model",
            type=str,
            metavar="STR",
            help="XLM model to use for initializing transformer encoder and/or decoder",
        )
        parser.add_argument(
            "--init-encoder-only",
            action="store_true",
            help="if set, don't load the XLM weights and embeddings into decoder",
        )
        parser.add_argument(
            "--init-decoder-only",
            action="store_true",
            help="if set, don't load the XLM weights and embeddings into encoder",
        )
        parser.add_argument("--latent-dimension",
            type=int,
            default=128,
            help="dimension of latent variables"
        )
        parser.add_argument("--latent-sample-size",
            type=int,
            default=20,
            help="latent variable sample size for computing expectation"
        )


    def __init__(self, args, encoder, decoder, p_model):
        super().__init__(args, encoder, decoder)
        self.latent_dim = self.args.latent_dimension
        self.latent_sample_size = self.args.latent_sample_size

        self.mean_layer1 = nn.Linear(self.args.encoder_embed_dim, self.latent_dim)
        self.mean_layer2 = nn.Linear(self.latent_dim, self.latent_dim)
        self.variance_layer1 = nn.Linear(self.args.encoder_embed_dim, self.latent_dim)
        self.variance_layer2 = nn.Linear(self.latent_dim, self.latent_dim)
        self.mapping_layer = nn.Linear(self.args.encoder_embed_dim+self.latent_dim, self.args.encoder_embed_dim)

        self.p_model = p_model

    @classmethod
    def build_model(self, args, task, cls_dictionary=MaskedLMDictionary):
        assert hasattr(args, "pretrained_esm_model"), (
            "You must specify a path for --pretrained-esm-model to use "
            "--arch transformer_from_pretrained_xlm"
        )
        return super().build_model(args, task)

    # @classmethod
    # def build_encoder(cls, args, src_dict, embed_tokens):
    #     return TransformerEncoderFromPretrainedESM(args, src_dict, embed_tokens)

    def forward(
        self,
        src_tokens,
        src_lengths,
        prev_output_tokens,
        return_all_hiddens: bool = True,
        features_only: bool = False,
        alignment_layer: Optional[int] = None,
        alignment_heads: Optional[int] = None,
    ):
        """
        Run the forward pass for an encoder-decoder model.

        Copied from the base class, but without ``**kwargs``,
        which are not supported by TorchScript.
        """
        encoder_outs = self.encoder(
            src_tokens, src_lengths=src_lengths, return_all_hiddens=return_all_hiddens
        )
        encoder_out = encoder_outs["encoder_out"][0]  # [src len, batch, dim]
        cls_encoder_out = encoder_out[0]  # [batch, dim]
        z_mean = self.mean_layer2(F.relu(self.mean_layer1(cls_encoder_out)))  # [batch, dim]
        z_std = torch.exp(self.variance_layer2(F.relu(self.variance_layer1(cls_encoder_out))))  # [batch, dim]

        total_decoder_out = []
        p_model_out = []
        for i in range(self.latent_sample_size):
            noise = torch.normal(mean=torch.zeros([z_mean.size()[0], z_mean.size()[1]]), std=1.0).half().to("cuda")
            latent = z_mean + z_std * noise
            latent = latent.unsqueeze(0)   # [1, batch, latent]
            latent = latent.expand(encoder_out.size()[0], latent.size(1), latent.size()[2])

            temp_encoder_out = torch.cat((encoder_out, latent), 2)
            this_encoder_out = self.mapping_layer(temp_encoder_out)
            encoder_outs["encoder_out"] = [this_encoder_out]

            decoder_out = self.decoder(
                prev_output_tokens,
                encoder_out=encoder_outs,
                features_only=features_only,
                alignment_layer=alignment_layer,
                alignment_heads=alignment_heads,
                src_lengths=src_lengths,
                return_all_hiddens=return_all_hiddens,
            )
            total_decoder_out.append(decoder_out)

            p_model_decoder_out = self.p_model.decoder(
                prev_output_tokens,
                encoder_out=encoder_outs,
                features_only=features_only,
                alignment_layer=alignment_layer,
                alignment_heads=alignment_heads,
                src_lengths=src_lengths,
                return_all_hiddens=return_all_hiddens,
            )
            p_model_out.append(p_model_decoder_out)

        return total_decoder_out, p_model_out, z_mean, z_std


@register_model_architecture("variational_is", "variational_is")
def base_architecture(args):
    transformer_base_architecture(args)


@register_model_architecture("variational_is", "variational_is_base")
def transformer_vae_base(args):
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 256)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 512)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 8)
    args.encoder_layers = getattr(args, "encoder_layers", 3)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", 256)
    args.decoder_ffn_embed_dim = getattr(args, "decoder_ffn_embed_dim", 512)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 8)
    args.decoder_layers = getattr(args, "decoder_layers", 3)
    base_architecture(args)


@register_model_architecture("variational_is", "variational_is_tiny")
def transformer_vae_base(args):
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 256)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 512)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 8)
    args.encoder_layers = getattr(args, "encoder_layers", 2)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", 256)
    args.decoder_ffn_embed_dim = getattr(args, "decoder_ffn_embed_dim", 512)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 8)
    args.decoder_layers = getattr(args, "decoder_layers", 2)
    base_architecture(args)
