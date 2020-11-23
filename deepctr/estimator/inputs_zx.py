import tensorflow as tf
import os

def input_fn_pandas(df, features, label=None, batch_size=256, num_epochs=1, shuffle=False, queue_capacity_factor=10,
                    num_threads=1):
    if label is not None:
        y = df[label]
    else:
        y = None
    if tf.__version__ >= "2.0.0":
        return tf.compat.v1.estimator.inputs.pandas_input_fn(df[features], y, batch_size=batch_size,
                                                             num_epochs=num_epochs,
                                                             shuffle=shuffle,
                                                             queue_capacity=batch_size * queue_capacity_factor,
                                                             num_threads=num_threads)

    return tf.estimator.inputs.pandas_input_fn(df[features], y, batch_size=batch_size, num_epochs=num_epochs,
                                               shuffle=shuffle, queue_capacity=batch_size * queue_capacity_factor,
                                               num_threads=num_threads)


def input_fn_tfrecord(file_path,compression_type, feature_description, label=None, batch_size=256, num_epochs=1, num_parallel_calls=8,
                      shuffle_factor=10, prefetch_factor=1,
                      ):
    def _get_file_list(file_path):
        if isinstance(file_path, list):
            all_files = path
        else:
            is_file = os.path.isfile(file_path)
            is_dir = os.path.isdir(file_path)
            if not is_file and not is_dir:
                raise Exception('%s is neither a file nor a dir' % file_path)
            if is_file:
                all_files = [file_path]
            else:
                files_temp=os.listdir(file_path)
                all_files =[file_path+f for f in files_temp] 

        files = [f for f in all_files if ('.%s' % 'tfrecord') in f]
        if len(files) == 0:
            raise Exception('No valid tfrecord files found in path: %s' % (file_path))
        return files
    def _parse_examples(serial_exmp):
        try:
            features = tf.parse_single_example(serial_exmp, features=feature_description)
        except AttributeError:
            features = tf.io.parse_single_example(serial_exmp, features=feature_description)
        if label is not None:
            labels = features.pop(label)
            return features, labels
        return features
    def input_fn():
        files=_get_file_list(file_path)
        dataset = tf.data.Dataset.list_files(files, shuffle=True)
        dataset = dataset.shuffle(100) #存入到buff中
        dataset=dataset.interleave(
            lambda filename:tf.data.TFRecordDataset(filename,compression_type=compression_type),
            cycle_length=5,
            num_parallel_calls=tf.data.experimental.AUTOTUNE)

        dataset = dataset.map(_parse_examples, num_parallel_calls=num_parallel_calls)
        if shuffle_factor > 0:
            dataset = dataset.shuffle(buffer_size=batch_size * shuffle_factor)

        dataset = dataset.repeat(num_epochs).batch(batch_size)

        if prefetch_factor > 0:
            dataset = dataset.prefetch(buffer_size=batch_size * prefetch_factor)
        try:
            iterator = dataset.make_one_shot_iterator()
        except AttributeError:
            iterator = tf.compat.v1.data.make_one_shot_iterator(dataset)
        return iterator.get_next()
    return input_fn

    
