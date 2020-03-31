import tensorflow as tf


def get_class_probs(x, c, l, w, params):
    """
    TODO: vectorize over l
    Return class probabilities for a single training example, based on the embedding vector and a vector of gaussian kernel
    centres c, one for each training sample, except the current sample
    labels: vector with labels associated with class centres c
    w: weights of each sample
    Dim(c) = (m - 1, embedding_dimension)
    Dim(l) = (m - 1,)
    Dim(w) = (m - 1,)
    Dim(x) = (embedding_dimension,)
    Dim(x) = (batch_size, embedding_dimension) -> not yet implemented
    """
    # print(x.shape)
    # print(c.shape)
    # print(l.shape)
    # print(w.shape)
    assert x.shape == (params['embedding_dim'],)
    assert c.shape[0] == l.shape[0]

    # if w is None:
    #     w = tf.ones_like(l)

    # p = tf.zeros((params.num_classes,))

    denominator = tf.reduce_sum(w * (tf.exp(-((x - c) ** 2) / (2 * params['global_scale'] ** 2))))
    p = []
    for class_i in range(params['num_classes']):
        c_match = c[l == class_i]
        w_match = w[l == class_i]
        enumerator = tf.reduce_sum(w_match * (tf.exp(-((x - c_match) ** 2) / (2 * params['global_scale'] ** 2))))
        p.append(enumerator / denominator)

    p = tf.concat([p], axis=0)

    p += 1e-6  # for numeric stability

    return p


def loss_fn(p, t, num_classes):
    """
    Loss of predictions
    dim(p): (num_classes,)
    t: integer, index of p
    """
    assert p.shape == (num_classes,)
    return -tf.math.log(p[t])