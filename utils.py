import tensorflow as tf


def get_nca_loss(x, l, nn_c, nn_l, w, params):
    """
    Return NCA loss for a single training example, based on the embedding vector, and its nearest neighbours
    (i.e., vectors of gaussian kernel centres, except the current sample).
    x: embedding vector of single training sample
    l: class label associated with embedding vector
    nn_c: vectors of gaussian kernel centres (nearest neighbours of x)
    nn_l: class labels associated with nearest neighbours
    w: weights of training sample
    Dim(c) = (k - 1, embedding_dimension)
    Dim(l) = (k - 1,)
    Dim(w) = (k - 1,)
    Dim(x) = (embedding_dimension,)
    """
    assert x.shape == (params['embedding_dim'],)
    assert nn_c.shape[0] == nn_l.shape[0]

    # class probability
    nn_w = tf.expand_dims(w, axis=-1)   # Dim(nn_w) = (k - 1, 1)
    denominator = tf.reduce_sum(nn_w * (tf.exp(-((x - nn_c) ** 2) / (2 * params['global_scale'] ** 2))))

    c_match = nn_c[l.numpy() == nn_l]
    w_match = nn_w[l.numpy() == nn_l]
    enumerator = tf.reduce_sum(w_match * (tf.exp(-((x - c_match) ** 2) / (2 * params['global_scale'] ** 2))))

    p = enumerator/denominator
    p += 1e-6  # for numeric stability

    # NCA loss
    loss = -tf.math.log(p)

    return loss
