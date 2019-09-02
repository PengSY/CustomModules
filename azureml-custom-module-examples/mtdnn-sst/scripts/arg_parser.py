import argparse


def preprocess_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, default=None, required=True)
    parser.add_argument("--output_dir", type=str, default=None, required=True)
    parser.add_argument("--do_lower_case", type=boolean_string, default=True)
    parser.add_argument("--model", type=str, default="bert-base-uncased")
    parser.add_argument("--max_seq_len", type=int, default=512)

    return parser


def train_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument("--train_data_dir", type=str, default=None, required=True)
    parser.add_argument("--init_checkpoint_dir", type=str, default=None, required=True)
    parser.add_argument("--output_dir", type=str, default=None, required=True)
    parser.add_argument("--max_seq_len", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--cuda", type=boolean_string, default=True)
    parser.add_argument('--multi_gpu_on', type=boolean_string, default=True)
    parser.add_argument("--dropout_w", type=float, default=0.000)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument('--update_bert_opt', default=0, type=int)
    parser.add_argument('--bert_dropout_p', type=float, default=0.1)
    parser.add_argument('--answer_opt', type=int, default=0, help='0,1')
    parser.add_argument('--label_size', type=str, default='2')
    parser.add_argument('--dropout_p', type=float, default=0.1)
    parser.add_argument('--vb_dropout', type=boolean_string, default=True)
    parser.add_argument('--init_ratio', type=float, default=1)
    parser.add_argument('--answer_dropout_p', type=float, default=0.1)
    parser.add_argument('--optimizer', type=str, default='adamax')
    parser.add_argument('--learning_rate', type=float, default=5e-5)
    parser.add_argument('--weight_decay', type=float, default=0)
    parser.add_argument('--warmup', type=float, default=0.1)
    parser.add_argument('--grad_clipping', type=float, default=0)
    parser.add_argument('--warmup_schedule', type=str, default='warmup_linear')
    parser.add_argument('--fp16', type=boolean_string, default=False)
    parser.add_argument('--fp16_opt_level', type=str, default='O1')
    parser.add_argument('--scheduler_type', type=str, default='ms', help='ms/rop/exp')
    parser.add_argument('--lr_gamma', type=float, default=0.5)
    parser.add_argument('--multi_step_lr', type=str, default='10,20,30')
    parser.add_argument('--ema_opt', type=int, default=0)
    parser.add_argument('--grad_accumulation_step', type=int, default=1)
    parser.add_argument('--global_grad_clipping', type=float, default=1.0)

    return parser


def score_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument("--test_data_dir", type=str, default=None, required=True)
    parser.add_argument("--trained_model_dir", type=str, default=None, required=True)
    parser.add_argument("--output_dir", type=str, default=None, required=True)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--cuda", type=boolean_string, default=False)
    parser.add_argument('--multi_gpu_on', type=boolean_string, default=True)

    return parser


def boolean_string(s):
    if s not in {"False", "True"}:
        raise ValueError("Not a valid boolean string")
    return s == "True"
