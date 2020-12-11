import tensorflow as tf
from tensorflow.python.keras.initializers import RandomNormal, Zeros
from tensorflow.python.feature_column.feature_column import _EmbeddingColumn
from collections import namedtuple, OrderedDict

DEFAULT_GROUP_NAME = "default_group"
class SparseFeat(namedtuple('SparseFeat',
                            ['name', 'vocabulary_size', 'embedding_dim', 'use_hash', 'dtype', 'embeddings_initializer',
                             'embedding_name',
                             'group_name', 'trainable'])):
    __slots__ = ()

    def __new__(cls, name, vocabulary_size, embedding_dim=4, use_hash=False, dtype="int32", embeddings_initializer=None,
                embedding_name=None,
                group_name=DEFAULT_GROUP_NAME, trainable=True):

        if embedding_dim == "auto":
            embedding_dim = 6 * int(pow(vocabulary_size, 0.25))
        if embeddings_initializer is None:
            embeddings_initializer = RandomNormal(mean=0.0, stddev=0.0001, seed=2020)

        if embedding_name is None:
            embedding_name = name

        return super(SparseFeat, cls).__new__(cls, name, vocabulary_size, embedding_dim, use_hash, dtype,
                                              embeddings_initializer,
                                              embedding_name, group_name, trainable)

    def __hash__(self):
        return self.name.__hash__()


class VarLenSparseFeat(namedtuple('VarLenSparseFeat',
                                  ['sparsefeat', 'maxlen', 'combiner', 'length_name', 'weight_name', 'weight_norm'])):
    __slots__ = ()

    def __new__(cls, sparsefeat, maxlen, combiner="mean", length_name=None, weight_name=None, weight_norm=True):
        return super(VarLenSparseFeat, cls).__new__(cls, sparsefeat, maxlen, combiner, length_name, weight_name,
                                                    weight_norm)

    @property
    def name(self):
        return self.sparsefeat.name

    @property
    def vocabulary_size(self):
        return self.sparsefeat.vocabulary_size

    @property
    def embedding_dim(self):
        return self.sparsefeat.embedding_dim

    @property
    def use_hash(self):
        return self.sparsefeat.use_hash

    @property
    def dtype(self):
        return self.sparsefeat.dtype

    @property
    def embeddings_initializer(self):
        return self.sparsefeat.embeddings_initializer

    @property
    def embedding_name(self):
        return self.sparsefeat.embedding_name

    @property
    def group_name(self):
        return self.sparsefeat.group_name

    @property
    def trainable(self):
        return self.sparsefeat.trainable

    def __hash__(self):
        return self.name.__hash__()


class DenseFeat(namedtuple('DenseFeat', ['name', 'dimension', 'dtype'])):
    __slots__ = ()

    def __new__(cls, name, dimension=1, dtype="float32"):
        return super(DenseFeat, cls).__new__(cls, name, dimension, dtype)

    def __hash__(self):
        return self.name.__hash__()

    # def __eq__(self, other):
    #     if self.name == other.name:
    #         return True
    #     return False

    # def __repr__(self):
    #     return 'DenseFeat:'+self.name



from .utils import LINEAR_SCOPE_NAME, variable_scope, get_collection, get_GraphKeys, input_layer, get_losses

def get_feature_names(feature_columns):
    features = build_input_features(feature_columns)
    return list(features.keys())

def build_input_features(feature_columns, prefix=''):
    input_features = OrderedDict()
    for fc in feature_columns:
        if isinstance(fc, SparseFeat):
            input_features[fc.name] = Input(
                shape=(1,), name=prefix + fc.name, dtype=fc.dtype)
        elif isinstance(fc, DenseFeat):
            input_features[fc.name] = Input(
                shape=(fc.dimension,), name=prefix + fc.name, dtype=fc.dtype)
        elif isinstance(fc, VarLenSparseFeat):
            input_features[fc.name] = Input(shape=(fc.maxlen,), name=prefix + fc.name,
                                            dtype=fc.dtype)
            if fc.weight_name is not None:
                input_features[fc.weight_name] = Input(shape=(fc.maxlen, 1), name=prefix + fc.weight_name,
                                                       dtype="float32")
            if fc.length_name is not None:
                input_features[fc.length_name] = Input((1,), name=prefix + fc.length_name, dtype='int32')

        else:
            raise TypeError("Invalid feature column type,got", type(fc))

    return input_features

def linear_model(features, linear_feature_columns):
    if tf.__version__ >= '2.0.0':
        linear_logits = tf.compat.v1.feature_column.linear_model(features, linear_feature_columns)
    else:
        linear_logits = tf.feature_column.linear_model(features, linear_feature_columns)
    return linear_logits


def get_linear_logit(features, linear_feature_columns, l2_reg_linear=0):
    with variable_scope(LINEAR_SCOPE_NAME):
        if not linear_feature_columns:
            linear_logits = tf.Variable([[0.0]], name='bias_weights')
        else:

            linear_logits = linear_model(features, linear_feature_columns)

            if l2_reg_linear > 0:
                for var in get_collection(get_GraphKeys().TRAINABLE_VARIABLES, LINEAR_SCOPE_NAME)[:-1]:
                    get_losses().add_loss(l2_reg_linear * tf.nn.l2_loss(var, name=var.name.split(":")[0] + "_l2loss"),
                                          get_GraphKeys().REGULARIZATION_LOSSES)
    return linear_logits


def input_from_feature_columns(features, feature_columns, l2_reg_embedding=0.0):
    dense_value_list = []
    sparse_emb_list = []
    for feat in feature_columns:
        if is_embedding(feat):
            sparse_emb = tf.expand_dims(input_layer(features, [feat]), axis=1)
            sparse_emb_list.append(sparse_emb)
            if l2_reg_embedding > 0:
                get_losses().add_loss(l2_reg_embedding * tf.nn.l2_loss(sparse_emb, name=feat.name + "_l2loss"),
                                      get_GraphKeys().REGULARIZATION_LOSSES)

        else:
            dense_value_list.append(input_layer(features, [feat]))

    return sparse_emb_list, dense_value_list


def is_embedding(feature_column):
    try:
        from tensorflow.python.feature_column.feature_column_v2 import EmbeddingColumn
    except:
        EmbeddingColumn = _EmbeddingColumn
    return isinstance(feature_column, (_EmbeddingColumn, EmbeddingColumn))
