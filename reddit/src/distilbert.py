# coding=utf-8
# Copyright 2019-present, the HuggingFace Inc. team, The Google AI Language Team and Facebook, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
 TF 2.0 DistilBERT model
"""

import warnings

import tensorflow as tf

from transformers.activations_tf import get_tf_activation
from transformers.file_utils import (
    MULTIPLE_CHOICE_DUMMY_INPUTS,
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
)
from transformers.modeling_tf_outputs import (
    TFBaseModelOutput,
    TFMaskedLMOutput,
    TFMultipleChoiceModelOutput,
    TFQuestionAnsweringModelOutput,
    TFSequenceClassifierOutput,
    TFTokenClassifierOutput,
)
from transformers.modeling_tf_utils import (
    TFMaskedLanguageModelingLoss,
    TFMultipleChoiceLoss,
    TFPreTrainedModel,
    TFQuestionAnsweringLoss,
    TFSequenceClassificationLoss,
    TFTokenClassificationLoss,
    get_initializer,
    input_processing,
    keras_serializable,
    shape_list,
)
from transformers.utils import logging
from transformers import DistilBertConfig
from .klayers import MultiHeadAttentionNoSoftmax
from transformers.models.distilbert.modeling_tf_distilbert import TFFFN     


logger = logging.get_logger(__name__)

_CHECKPOINT_FOR_DOC = "distilbert-base-uncased"
_CONFIG_FOR_DOC = "DistilBertConfig"
_TOKENIZER_FOR_DOC = "DistilBertTokenizer"

TF_DISTILBERT_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "distilbert-base-uncased",
    "distilbert-base-uncased-distilled-squad",
    "distilbert-base-cased",
    "distilbert-base-cased-distilled-squad",
    "distilbert-base-multilingual-cased",
    "distilbert-base-uncased-finetuned-sst-2-english",
    # See all DistilBERT models at https://huggingface.co/models?filter=distilbert
]


class CTXTransformerBlock(tf.keras.layers.Layer):
    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)

        self.n_heads = config.n_heads
        self.dim = config.dim
        self.hidden_dim = config.hidden_dim
        self.dropout = tf.keras.layers.Dropout(config.dropout)
        self.activation = config.activation
        self.output_attentions = config.output_attentions

        assert (
            config.dim % config.n_heads == 0
        ), f"Hidden size {config.dim} not dividable by number of heads {config.n_heads}"

        self.attention = MultiHeadAttentionNoSoftmax(num_heads=self.n_heads, 
                                                     key_dim=768, 
                                                     attention_axes=(0,1))
        self.sa_layer_norm = tf.keras.layers.LayerNormalization(epsilon=1e-12, name="sa_layer_norm")
        self.ffn = TFFFN(config, name="ffn")
        self.output_layer_norm = tf.keras.layers.LayerNormalization(epsilon=1e-12, name="output_layer_norm")

    def call(self, x, context):  # removed: src_enc=None, src_len=None
        """
        Parameters:
            x: target
            context: context
        """
        # Self-Attention
        #sa_output = self.attention(x, x, x, attn_mask, head_mask, output_attentions, training=training)
        sa_output = self.attention(query=x, value=context)
        sa_output = sa_output[0] # weighted contexts
        sa_output = self.sa_layer_norm(sa_output + x)  # ?

        # Feed Forward Network
        ffn_output = self.ffn(sa_output)  # ?
        ffn_output = self.output_layer_norm(ffn_output + sa_output) # ?

        output = (ffn_output,)
        return output
