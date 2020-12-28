# coding=utf-8
"""
模型构建类
illool@163.com
QQ:122018919
"""
import tensorflow as tf
import bert.modeling
import bert.optimization
from tensorflow.contrib import rnn
import tensorflow.contrib.seq2seq as seq2seq


def get_en_decoder_cell(rnn_size, num_layers, keep_prob):
    gru_stack = [rnn.GRUCell(num_units=rnn_size) for _ in range(num_layers)]
    lstm_cell = rnn.MultiRNNCell(gru_stack)
    mlstm_cell = rnn.DropoutWrapper(cell=lstm_cell, input_keep_prob=1.0, output_keep_prob=keep_prob)
    # 用全零来初始化state
    # init_state = mlstm_cell.zero_state(batch_size, dtype=tf.float32)
    return mlstm_cell


def decoding_layer_train(encoder_state, dec_cell, lable_ids_seq_embedding,
                         actual_len_batch, max_seq_len_batch,
                         output_layer):
    with tf.variable_scope("decode_train"):
        training_helper = seq2seq.TrainingHelper(inputs=lable_ids_seq_embedding, sequence_length=actual_len_batch,
                                                 time_major=False)
        training_decoder = seq2seq.BasicDecoder(dec_cell, training_helper, encoder_state, output_layer)
        training_decoder_output, _, _ = seq2seq.dynamic_decode(training_decoder, impute_finished=True,
                                                               maximum_iterations=max_seq_len_batch)
    # tf.contrib.seq2seq.BasicDecoderOutput
    # final_outputs是一个namedtuple，里面包含两项(rnn_outputs, sample_id)
    # rnn_output: [batch_size, decoder_targets_length, vocab_size]，保存decode每个时刻每个单词的概率，可以用来计算loss
    # sample_id: [batch_size], tf.int32，保存最终的编码结果。可以表示最后的答案
    return training_decoder_output


def decoding_layer_infer(encoder_state, dec_cell, dec_embeddings, start_of_sequence_id,
                         end_of_sequence_id, max_seq_len_batch,
                         output_layer, batch_size):
    with tf.variable_scope("decode_infer"):
        start_tokens = tf.tile(tf.constant([start_of_sequence_id], dtype=tf.int32), [batch_size])
        helper = seq2seq.GreedyEmbeddingHelper(dec_embeddings, start_tokens, end_of_sequence_id)
        decoder = seq2seq.BasicDecoder(dec_cell, helper, encoder_state, output_layer=output_layer)
        dec_outputs, dec_state, dec_sequence_length = \
            seq2seq.dynamic_decode(decoder, impute_finished=True, maximum_iterations=max_seq_len_batch)
    # tf.contrib.seq2seq.BasicDecoderOutput
    # final_outputs是一个namedtuple，里面包含两项(rnn_outputs, sample_id)
    # rnn_output: [batch_size, decoder_targets_length, vocab_size]，保存decode每个时刻每个单词的概率，可以用来计算loss
    # sample_id: [batch_size], tf.int32，保存最终的编码结果。可以表示最后的答案
    return dec_outputs


def decoding_layer(dec_input, encoder_state,
                   target_sequence_length, max_target_sequence_length,
                   rnn_size,
                   num_layers, labels, num_labels,
                   batch_size, keep_prob, decoding_embedding_size):
    # embedding target sequence
    # dec_embeddings = tf.one_hot(labels, depth=num_labels, dtype=tf.float32)
    dec_embeddings = tf.Variable(tf.random_uniform([num_labels, decoding_embedding_size]))
    dec_embed_input = tf.nn.embedding_lookup(dec_embeddings, dec_input)
    # construct decoder lstm cell
    dec_cell = get_en_decoder_cell(rnn_size, num_layers, keep_prob)
    # create output layer to map the outputs of the decoder to the elements of our lable
    output_layer = tf.layers.Dense(num_labels,
                                   kernel_initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1))
    # decoder train
    dec_outputs_train = decoding_layer_train(encoder_state, dec_cell, dec_embed_input,
                                             target_sequence_length, max_target_sequence_length,
                                             output_layer)
    # decoder inference
    start_of_sequence_id = labels.index('[CLS]')
    end_of_sequence_id = labels.index('[SEP]')
    dec_outputs_infer = decoding_layer_infer(encoder_state, dec_cell, dec_embeddings, start_of_sequence_id,
                                             end_of_sequence_id, max_target_sequence_length, output_layer, batch_size)
    return dec_outputs_train, dec_outputs_infer


def encoding_layer(rnn_size, layer_size, input_x_embedded, keep_prob):
    # define encoder
    with tf.variable_scope('encoder'):
        encoder = get_en_decoder_cell(rnn_size, layer_size, keep_prob)

    encoder_outputs, encoder_state = tf.nn.dynamic_rnn(encoder, input_x_embedded, dtype=tf.float32)
    return encoder_outputs, encoder_state


def create_model(bert_config, is_training, input_ids, input_mask, segment_ids,
                 labels, num_labels, use_one_hot_embeddings, max_seq_length, batch_size, actual_length, label_list):
    """Creates a classification model."""
    model = bert.modeling.BertModel(
        config=bert_config,
        is_training=is_training,
        input_ids=input_ids,
        input_mask=input_mask,
        token_type_ids=segment_ids,
        use_one_hot_embeddings=use_one_hot_embeddings)

    # labels.shape,(32, 128),input_ids.shape, (32, 128), type(labels),<class 'tensorflow.python.framework.ops.Tensor'>
    output_layer = model.get_sequence_output()  # (64, 128, 768)
    rnn_size = output_layer.shape[-1].value
    num_layers = 3
    keep_prob = 0.9
    decoding_embedding_size = int(rnn_size/6)

    with tf.variable_scope("loss"):
        labels = tf.sparse_to_dense(labels.indices, labels.dense_shape, labels.values)
        encoder_outputs, encoder_state = encoding_layer(rnn_size, num_layers, output_layer, keep_prob)
        outputs_train, outputs_infer = decoding_layer(labels, encoder_state,
                                                      actual_length, max_seq_length,
                                                      rnn_size,
                                                      num_layers, label_list, num_labels,
                                                      batch_size, keep_prob, decoding_embedding_size)
    return outputs_train, outputs_infer


def model_fn_builder(bert_config, num_labels, init_checkpoint, learning_rate,
                     num_train_steps, num_warmup_steps, use_tpu,
                     use_one_hot_embeddings, label_list):
    """Returns `model_fn` closure for TPUEstimator."""

    def model_fn(features, labels, mode, params):  # pylint: disable=unused-argument
        """The `model_fn` for TPUEstimator."""

        tf.logging.info("*** Features ***")
        for name in sorted(features.keys()):
            if name != "label_ids":
                tf.logging.info("  name = %s, shape = %s" % (name, features[name].shape))
            else:
                tf.logging.info("  name = %s, shape = %s" % (name, features[name].dense_shape))

        input_ids = features["input_ids"]
        input_mask = features["input_mask"]
        segment_ids = features["segment_ids"]
        label_ids = features["label_ids"]
        actual_length = features["actual_length"]
        is_real_example = None
        if "is_real_example" in features:
            is_real_example = tf.cast(features["is_real_example"], dtype=tf.float32)
        else:
            is_real_example = tf.ones(tf.shape(label_ids), dtype=tf.float32)

        is_training = (mode == tf.estimator.ModeKeys.TRAIN)

        (outputs_train, outputs_infer) = create_model(
            bert_config, is_training, input_ids, input_mask, segment_ids, label_ids,
            num_labels, use_one_hot_embeddings, params["max_seq_length"], params["batch_size"],
            actual_length, label_list)

        tvars = tf.trainable_variables()
        initialized_variable_names = {}
        if init_checkpoint:  # 判断是否是初次训练，要是断点训练init_checkpoint设置为None
            (assignment_map, initialized_variable_names
             ) = bert.modeling.get_assignment_map_from_checkpoint(tvars, init_checkpoint)
            tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
        tf.logging.info("**** Trainable Variables ****")
        for var in tvars:
            init_string = ""
            if var.name in initialized_variable_names:
                init_string = ", *INIT_FROM_CKPT*"
            tf.logging.info("  name = %s, shape = %s%s", var.name, var.shape,
                            init_string)

        if mode == tf.estimator.ModeKeys.TRAIN:
            training_logits = tf.identity(outputs_train.rnn_output, 'logits')
            masks = tf.sequence_mask(actual_length, params["max_seq_length"], dtype=tf.float32, name="masks")
            targets = tf.sparse_to_dense(label_ids.indices, label_ids.dense_shape, label_ids.values,
                                         default_value=num_labels - 1)
            # targets = tf.boolean_mask(targets, masks)
            # targets = tf.reshape(targets, [outputs_train.rnn_output.shape[0], -1])
            # current_ts = tf.to_int32(tf.minimum(tf.shape(targets)[1], tf.shape(training_logits)[1]))
            # targets = tf.slice(targets, begin=[0, 0, 0], size=[-1, current_ts, -1])

            cost = seq2seq.sequence_loss(
                training_logits,
                targets,
                masks,
                average_across_batch=True
            )

            optimizer = tf.train.AdamOptimizer(learning_rate)

            # minimize函数用于添加操作节点，用于最小化loss，并更新var_list.
            # 该函数是简单的合并了compute_gradients()与apply_gradients()函数返回为一个优化更新后的var_list，
            # 如果global_step非None，该操作还会为global_step做自增操作

            # 这里将minimize拆解为了以下两个部分：

            # 对var_list中的变量计算loss的梯度 该函数为函数minimize()的第一部分，返回一个以元组(gradient, variable)组成的列表
            gradients = optimizer.compute_gradients(cost)
            capped_gradients = [(tf.clip_by_value(grad, -5., 5.), var) for grad, var in gradients if grad is not None]
            # 将计算出的梯度应用到变量上，是函数minimize()的第二部分，返回一个应用指定的梯度的操作Operation，对global_step做自增操作
            train_op = optimizer.apply_gradients(capped_gradients)

            # tvars = tf.trainable_variables()
            # grads, _ = tf.clip_by_global_norm(tf.gradients(cost, tvars), 1.)
            # optimizer = tf.train.AdamOptimizer(1e-3)
            # train_op = optimizer.apply_gradients(zip(grads, tvars))

            # train_op = bert.optimization.create_optimizer(
            #     cost, learning_rate, num_train_steps, num_warmup_steps, use_tpu)

            output_spec = tf.estimator.EstimatorSpec(
                mode=mode,
                # For mode==ModeKeys.TRAIN: 需要的参数是 loss and train_op.
                loss=cost,
                train_op=train_op,
            )
            return output_spec
        elif mode == tf.estimator.ModeKeys.EVAL:

            def metric_fn(cost_, label_ids_, decoded_, is_real_example_):
                # predictions = tf.argmax(logits, axis=-1, output_type=tf.int32)
                decoded_dense_, = tf.sparse_to_dense(decoded_[0].indices, decoded_[0].dense_shape, decoded_[0].values)
                accuracy = tf.metrics.accuracy(
                    labels=label_ids_, predictions=decoded_dense_, weights=is_real_example_)
                loss = tf.metrics.mean(values=cost_, weights=is_real_example_)
                return {
                    "eval_accuracy": accuracy,
                    "eval_loss": loss,
                }

            targets = tf.reshape(label_ids, [-1])
            logits_flat = tf.reshape(outputs_train.rnn_output, [-1, num_labels])
            cost = tf.losses.sparse_softmax_cross_entropy(targets, logits_flat)

            eval_metrics = metric_fn(cost, label_ids, outputs_infer.sample_id, is_real_example)

            output_spec = tf.estimator.EstimatorSpec(
                mode=mode,
                # For mode==ModeKeys.EVAL:  需要的参数是 loss.
                loss=cost,
                eval_metric_ops=eval_metrics,
            )
            return output_spec
        elif mode == tf.estimator.ModeKeys.PREDICT:
            predicting_logits = tf.identity(outputs_infer.sample_id, name='predictions')
            # 将实际值和预测值生成字典
            predictions = {
                "input_ids": input_ids,
                "label_ids": tf.sparse_to_dense(label_ids.indices, label_ids.dense_shape, label_ids.values,
                                                default_value=num_labels-1),
                "predicts": predicting_logits
            }
            output_spec = tf.estimator.EstimatorSpec(
                mode=mode,
                # For mode==ModeKeys.PREDICT: 需要的参数是 predictions.
                predictions=predictions,
            )
            return output_spec
        else:
            raise ValueError(
                "Only TRAIN and PREDICT modes are supported: %s" % mode)

    return model_fn
