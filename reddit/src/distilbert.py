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

from transformers.models.distilbert.modeling_tf_distilbert import TFFFN, TFMultiHeadSelfAttention


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
        self.attention = TFMultiHeadSelfAttention(config, name="attention")

    def call(self, x, n_posts):
        """
        Parameters:
            x: tf.Tensor(bs, seq_length, dim)
            attn_mask: tf.Tensor(bs, seq_length)

        Outputs: sa_weights: tf.Tensor(bs, n_heads, seq_length, seq_length) The attention weights ffn_output:
        tf.Tensor(bs, seq_length, dim) The output of the transformer block contextualization.
        """
        sa_output = self.attention(x, x, x, 
                                   tf.constant(1, shape=[1,n_posts]), 
                                   head_mask=None, 
                                   output_attentions=False, 
                                   training=True)
        return sa_output[0][0,:,:] # 
