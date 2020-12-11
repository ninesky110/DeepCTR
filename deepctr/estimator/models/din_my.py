# -*- coding:utf-8 -*-
import tensorflow as tf

from ..feature_column import SparseFeat, VarLenSparseFeat, DenseFeat, build_input_features
from ..inputs import create_embedding_matrix, embedding_lookup,my_embedding_lookup, get_dense_input, varlen_embedding_lookup,get_varlen_pooling_list
from ...layers.core import DNN
from ...layers.sequence import AttentionSequencePoolingLayer
from ...layers.utils import concat_func, NoMask, combined_dnn_input
from ..utils import deepctr_model_fn, DNN_SCOPE_NAME, variable_scope

def my_DinEstimator(dnn_feature_columns, history_feature_list, dnn_use_bn=False,
                dnn_hidden_units=(200, 80), dnn_activation='relu', att_hidden_size=(80, 40), att_activation="dice",
                att_weight_normalization=False, l2_reg_dnn=0, l2_reg_embedding=1e-6, dnn_dropout=0, seed=1024,
                task='binary', model_dir=None, config=None,linear_optimizer='Ftrl',dnn_optimizer='Adagrad', training_chief_hooks=None):

    def _model_fn(features, labels, mode, config):
        train_flag = (mode == tf.estimator.ModeKeys.TRAIN)

        sparse_feature_columns = list(#获取稀疏特征对象
            filter(lambda x: isinstance(x, SparseFeat), dnn_feature_columns)) if dnn_feature_columns else []
        dense_feature_columns = list(#获取稠密特征对象
            filter(lambda x: isinstance(x, DenseFeat), dnn_feature_columns)) if dnn_feature_columns else []
        varlen_sparse_feature_columns = list(#获取序列特征对象
            filter(lambda x: isinstance(x, VarLenSparseFeat), dnn_feature_columns)) if dnn_feature_columns else []

        history_feature_columns = []#history_feature_list中包含的特征
        sparse_varlen_feature_columns = []#history_feature_list中不包含的特征
        history_fc_names = list(map(lambda x: "hist_" + x, history_feature_list))#获取序列特征名称
        for fc in varlen_sparse_feature_columns:
            feature_name = fc.name
            if feature_name in history_fc_names:
                history_feature_columns.append(fc)
            else:
                sparse_varlen_feature_columns.append(fc)

        inputs_list = list(features.values())
        #对离散特征生成embedding矩阵
        embedding_dict = create_embedding_matrix(dnn_feature_columns, l2_reg_embedding, seed, prefix="")

        # query_emb_list = embedding_lookup(embedding_dict, features, sparse_feature_columns, history_feature_list,
        #                                 history_feature_list, to_list=True)
        # keys_emb_list = embedding_lookup(embedding_dict, features, history_feature_columns, history_fc_names,
        #                                 history_fc_names, to_list=True)
        #获取离散特征的embedding编码表示
        dnn_input_emb_list = embedding_lookup(embedding_dict, features, sparse_feature_columns,
                                            mask_feat_list=history_feature_list, to_list=True)
        #获取dense特征数据
        dense_value_list = get_dense_input(features, dense_feature_columns)

        #将不进行attention的序列特征进行embedding编码
        sequence_embed_dict = varlen_embedding_lookup(embedding_dict, features, sparse_varlen_feature_columns)
        sequence_embed_list = get_varlen_pooling_list(sequence_embed_dict, features, sparse_varlen_feature_columns,
                                                    to_list=True)

        dnn_input_emb_list += sequence_embed_list

        # keys_emb = concat_func(keys_emb_list, mask=True)
        deep_input_emb = concat_func(dnn_input_emb_list)
        # query_emb = concat_func(query_emb_list, mask=True)
        keys_emb_list=my_embedding_lookup(embedding_dict,features,history_feature_columns,history_fc_names,history_fc_names)
        keys_emb_play=keys_emb_list['hist_play_channel_index']
        keys_emb_search_click=keys_emb_list['hist_search_click_channel_index']
        keys_emb_feeds_click=keys_emb_list['hist_feeds_click_channel_index']
        keys_emb_vip_play=keys_emb_list['hist_vip_play_channel_index']
        
        query_emb=my_embedding_lookup(embedding_dict, features, sparse_feature_columns, history_feature_list,history_feature_list)['channel_index']

        hist_play = AttentionSequencePoolingLayer(att_hidden_size, att_activation,
                                            weight_normalization=att_weight_normalization, supports_masking=True)([query_emb, keys_emb_play])
        hist_search_click = AttentionSequencePoolingLayer(att_hidden_size, att_activation,
                                            weight_normalization=att_weight_normalization, supports_masking=True)([query_emb, keys_emb_search_click])
        hist_feeds_click = AttentionSequencePoolingLayer(att_hidden_size, att_activation,
                                            weight_normalization=att_weight_normalization, supports_masking=True)([query_emb, keys_emb_feeds_click])
        hist_vip_play = AttentionSequencePoolingLayer(att_hidden_size, att_activation,
                                            weight_normalization=att_weight_normalization, supports_masking=True)([query_emb, keys_emb_vip_play])

        deep_input_emb = tf.keras.layers.Concatenate()([NoMask()(deep_input_emb), hist_play,hist_search_click,hist_feeds_click,hist_vip_play])
        deep_input_emb = tf.keras.layers.Flatten()(deep_input_emb)
        dnn_input = combined_dnn_input([deep_input_emb], dense_value_list)
        output = DNN(dnn_hidden_units, dnn_activation, l2_reg_dnn, dnn_dropout, dnn_use_bn, seed=seed)(dnn_input)
        final_logit = tf.keras.layers.Dense(1, use_bias=False,
                                        kernel_initializer=tf.keras.initializers.glorot_normal(seed))(output)

        return deepctr_model_fn(features, mode, final_logit, labels, task, linear_optimizer, dnn_optimizer,
                                training_chief_hooks
                                =training_chief_hooks)

    return tf.estimator.Estimator(_model_fn, model_dir=model_dir, config=config)
