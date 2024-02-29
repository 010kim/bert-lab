from absl import app
from absl import flags
from absl import logging

import modeling
import tokenization

FLAGS = flags.FLAGS

# Required parameters
flags.DEFINE_string('data_dir', None, 'The input data dir. Should contain the .tsv files for the task')
flags.DEFINE_string('bert_config_file', None,'The config json file for pre-trained BERT model')
flags.DEFINE_string('task_name', None, 'Task to train')
flags.DEFINE_string('vocab_file', None, 'Vocab file for BERT model')
flags.DEFINE_string('output_dir', None, 'Directory for model checkpoints')

# Other parameters
flags.DEFINE_string('init_checkpoint', None, 'Initial checkpoint from a pre-trained BERT model')
flags.DEFINE_bool('do_lower_case', True, 'Whether to lower case the input text. True for uncased models')
flags.DEFINE_integer('max_seq_length', 128, 'The maximum total input sequence length after WordPiece tokenization')
flags.DEFINE_bool('do_train', False, "whether to run training")
flags.DEFINE_bool('do_eval', False, 'Whether to run eval on the dev set')
flags.DEFINE_bool('do_predict', False, 'Whether to run the model in inference mode on the test')
flags.DEFINE_integer('train_batch_size', 32, 'Total batch size for training')
flags.DEFINE_integer('eval_batch_size', 8, 'Total batch size for eval')
flags.DEFINE_integer('predict_batch_size', 8, 'Total batch size for predict')
flags.DEFINE_float('learning_rate', 5e-5, 'The initial learning rate for Adam')
flags.DEFINE_float('num_train_epochs', 3.0, 'Total number of training epochs to perform')
flags.DEFINE_float('warmup_proportion', 0.1, 'Proportion of training to perform linear learning rate warm up')
flags.DEFINE_integer('save_checkpoints_steps', 1000, 'How often to save the model checkpoint')
flags.DEFINE_integer('iterations_per_loop', 1000, 'How many steps to make in each estimator call')
flags.DEFINE_bool('use_tpu', False, 'Whether to use TPU or GPU/CPU')
flags.DEFINE_string('tpu_name', None, 'cloud tpu to use for training')
flags.DEFINE_string('tpu_zone', None, '[Optional] GCE zone for TPU')
flags.DEFINE_string('gcp_project', None, '[Optional] Project name for tpu-enabled project')
flags.DEFINE_string('master', None, '[Optional] Tensorflow master URL')
flags.DEFINE_integer('num_tpu_cores', 8, 'Only used if `use_tpu` is True')


class DataProcessor():
    pass

class XnliProcessor(DataProcessor):
    pass


class MnliProcessor(DataProcessor):
    pass


class MrpcProcessor(DataProcessor):
    pass


class ColaProcessor(DataProcessor):
    pass


def main(argv):
    logging.set_verbosity(logging.INFO)
    processors = {
        'cola': ColaProcessor,
        'mnli': MnliProcessor,
        'mrpc': MrpcProcessor,
        'xnli': XnliProcessor,
    }

    tokenization.validate_case_matches_checkpoint(FLAGS.do_lower_case, FLAGS.init_checkpoint)
    if not FLAGS.do_train and not FLAGS.do_eval and not FLAGS.do_predict:
        raise ValueError("At least one of `do_train`, `do_eval` or `do_predict' must be True.")
    
    bert_config = modeling.BertConfig.from_json_file(FLAGS.bert_config_file)

    logging.info("PASS")

if __name__ == "__main__":
    flags.mark_flag_as_required('data_dir')
    flags.mark_flag_as_required('task_name')
    flags.mark_flag_as_required('vocab_file')
    flags.mark_flag_as_required('bert_config_file')
    flags.mark_flag_as_required('output_dir')
    app.run(main)
    