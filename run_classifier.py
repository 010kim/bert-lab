from absl import app
from absl import flags
from absl import logging

import csv
import modeling
import os
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


class InputExample():
    "A single training/test example for simple sequence classification"
    def __init__(self, guid, text_a, text_b=None, label=None):
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label


class DataProcessor():
    """Base class for sequence classification data sets"""

    def get_train_examples(self, data_dir):
        raise NotImplementedError
    
    def get_dev_examples(self, data_dir):
        raise NotImplementedError

    def get_test_examples(self, data_dir):
        raise NotImplementedError
    
    def get_labels(self):
        raise NotImplementedError
    
    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        with open(input_file, 'r') as f:
            reader = csv.reader(f, delimiter='\t', quotechar=quotechar)
            lines = []
            for line in reader:
                lines.append(line)
            return lines


class XnliProcessor(DataProcessor):
    pass


class MnliProcessor(DataProcessor):
    pass


class MrpcProcessor(DataProcessor):
    def get_train_examples(self, data_dir):
        return self._create_examples(self._read_tsv(os.path.join(data_dir, 'train.tsv')), 'train')
    
    def get_dev_examples(self, data_dir):
        return self._create_examples(self._read_tsv(os.path.join(data_dir, 'dev.tsv')), 'dev')

    def get_test_examples(self, data_dir):
        return self._create_examples(self._read_tsv(os.path.join(data_dir, 'test.tsv')), 'test')

    def get_labels(self):
        return ['0', '1']

    def _create_examples(self, lines, set_type):
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = f"{set_type}-{i}"
            text_a = tokenization.convert_to_unicode(line[3])
            text_b = tokenization.convert_to_unicode(line[4])
            if set_type == "test":
                label = "0"
            else:
                label = tokenization.convert_to_unicode(line[0])
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


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

    if FLAGS.max_seq_length > bert_config.max_position_embeddings:
        raise ValueError("Cannot use sequence length")

    if not os.path.isdir(FLAGS.output_dir):
        os.makedirs(FLAGS.output_dir)

    task_name = FLAGS.task_name.lower()
    
    if task_name not in processors:
        raise ValueError(f"Task not found: {task_name}")
    
    processor = processors[task_name]()
    label_list = processor.get_labels()
    
    tokenizer = tokenization.FullTokenizer(vocab_file=FLAGS.vocab_file, do_lower_case=FLAGS.do_lower_case)

    # finish tokenizer and skip tpu related part

    logging.info("PASS")

if __name__ == "__main__":
    flags.mark_flag_as_required('data_dir')
    flags.mark_flag_as_required('task_name')
    flags.mark_flag_as_required('vocab_file')
    flags.mark_flag_as_required('bert_config_file')
    flags.mark_flag_as_required('output_dir')
    app.run(main)
    