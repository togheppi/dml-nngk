import os
import numpy as np
from models import *
import argparse
from utils import *
from dataset import get_dataset
import hnswlib
from tensorboard.plugins import projector
# from tensorflow.compat.v1 import ConfigProto
# config = ConfigProto()
# config.gpu_options.allow_growth = True

parser = argparse.ArgumentParser()
parser.add_argument("--train_dir", default='/mnt/vision-nas/minjae/data-sets/celeba/sample', type=str)
parser.add_argument("--num_samples", default=59, type=int)
parser.add_argument("--image_size", default=224, type=int)
parser.add_argument("--embedding_dim", default=512, type=int)
parser.add_argument("--num_neighbors", default=10, type=int)
parser.add_argument("--global_scale", default=20.0, type=float)
parser.add_argument("--num_epochs", default=200, type=int)
parser.add_argument("--batch_size", default=8, type=int)
parser.add_argument("--lr", default=1e-5, type=float)
parser.add_argument("--c_update_interval_epoch", default=5, type=int)
parser.add_argument("--use_weights", default=True, type=bool)
parser.add_argument("--ckpt_dir", default='./ckpt', type=str)
parser.add_argument("--log_dir", default='./log', type=str)


# nn search using 3rd-party library (HNSW: https://github.com/nmslib/hnswlib)
def search_nns(data, k, epoch):
    num_elements, dim = data.shape

    # Declaring index
    p = hnswlib.Index(space='cosine', dim=dim)  # possible options are l2, cosine or ip

    # Initing index - the maximum number of elements should be known beforehand
    p.init_index(max_elements=num_elements, ef_construction=200, M=64)

    # Element insertion (can be called several times):
    p.add_items(data)

    # Controlling the recall by setting ef:
    p.set_ef(200)  # ef should always be > k

    # Query dataset, k - number of closest elements (returns 2 numpy arrays)
    neighbor_ids, distances = p.knn_query(data, k)

    # Serializing and deleting the index:
    index_path = 'nn_ids_list_%depoch.bin' % epoch
    print("Saving index to '%s'" % index_path)
    p.save_index(index_path)

    return neighbor_ids


def update_centres(model, init_ds, storage, epoch, k=5):
    for images, labels, ids, _ in init_ds:
        init_out = model(images, training=True)
        storage['centres'][ids.numpy()] = init_out['z']
        storage['labels'][ids.numpy()] = labels

    # search nns
    print('Searching neighbors...')
    neighbor_ids = search_nns(storage['centres'], k, epoch)

    return storage, neighbor_ids


def train_step(data, model, optimizer, storage, neighbor_ids, args):
    # data
    images, labels, ids = data

    with tf.GradientTape() as tape:
        # forward pass
        outs = model(images, training=True)
        outs_z = outs["z"]
        outs_w = outs["w"]

        # get indices, centres, labels of nearest neighbors
        nn_ids = neighbor_ids[ids.numpy()][:, 1:]  # exclude input sample itself from neighbor list
        nn_centres = storage['centres'][nn_ids]
        nn_labels = storage['labels'][nn_ids]

        # compute nca loss
        loss = 0.0
        for i, (out_z, l, nn_c, nn_l, out_w) in enumerate(zip(outs_z, labels, nn_centres, nn_labels, outs_w)):
            loss += get_nca_loss(out_z, l, nn_c, nn_l, out_w, args)

    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss


if __name__ == "__main__":
    # Load the parameters
    args = vars(parser.parse_args())

    # Model, Optimizer
    model = resnet18(args)
    optimizer = tf.optimizers.Adam(args['lr'])

    # metrics
    metric_loss = tf.keras.metrics.Mean('NCA loss', dtype=tf.float32)

    # Initiate storage for gaussian centres & neighbor list
    print('initializing centres..')
    init_ds = get_dataset(args['train_dir'], batch_size=args['batch_size'], epochs=1, shuffle=False)
    storage = {
        'centres': np.zeros((args['num_samples'], args['embedding_dim']), dtype=np.float32),
        'labels': np.zeros((args['num_samples']), dtype=np.int32),
    }
    neighbor_ids = np.zeros((args['num_samples'], args['num_neighbors']), dtype=np.int32)

    # Training data
    train_ds = get_dataset(args['train_dir'], batch_size=args['batch_size'], epochs=args['num_epochs'], shuffle=True)
    iters_per_epoch = args['num_samples'] // args['batch_size']
    c_update_interval_iter = args['c_update_interval_epoch'] * iters_per_epoch

    # prepare saver
    ckpt = tf.train.Checkpoint(model=model, optimizer=optimizer)
    manager = tf.train.CheckpointManager(ckpt, args['ckpt_dir'], max_to_keep=2)
    ckpt.restore(manager.latest_checkpoint)
    if manager.latest_checkpoint:
        print('Classifier: Continue training from {}'.format(manager.latest_checkpoint))
    else:
        print('Not restoring from saved checkpoint')

    # setup tensorboards
    train_summary_writer = tf.summary.create_file_writer(args['log_dir'])

    # Train starts
    for images, labels, ids, _ in train_ds:
        data = images, labels, ids

        # Get current step
        step = optimizer.iterations.numpy()
        epoch = step // iters_per_epoch

        # Update gaussian centres & k-nearest neighbor list
        if step == 0 or step % c_update_interval_iter == 0:
            print("Updating centres & neighbor list..".format(epoch))
            storage, neighbor_ids = update_centres(model, init_ds, storage, epoch, k=args['num_neighbors'])

        # Train step
        loss = train_step(data, model, optimizer, storage, neighbor_ids, args)

        # update metrics
        metric_loss(loss)

        # save to tensorboard
        with train_summary_writer.as_default():
            tf.summary.scalar('NCA loss', metric_loss.result(), step=step)

        if step % c_update_interval_iter == 0:
            manager.save(checkpoint_number=step)

            checkpoint = tf.train.Checkpoint(centres=tf.Variable(storage['centres'], name='centres'))
            checkpoint.save(os.path.join(args['log_dir'], "centres.ckpt"))

            # Save Labels separately on a line-by-line manner.
            with open(os.path.join(args['log_dir'], 'metadata.tsv'), "w") as f:
                for id in storage['labels']:
                    f.write("{}\n".format(id))

            # Set up config for projector
            config = projector.ProjectorConfig()
            embedding = config.embeddings.add()
            # # The name of the tensor will be suffixed by `/.ATTRIBUTES/VARIABLE_VALUE`
            embedding.tensor_name = "centres/.ATTRIBUTES/VARIABLE_VALUE"
            embedding.metadata_path = 'metadata.tsv'
            projector.visualize_embeddings(args['log_dir'], config)

        print('Epoch: [{}/{}], Loss: {:.4f}'.format(epoch + 1, args['num_epochs'], loss))

        if (epoch + 1) == args['num_epochs']:
            np.save('feature_centers.npy', storage['centres'])
            print('done.')
            break

