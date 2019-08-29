import pandas as pd
import pyarrow.parquet as pq  # noqa: F401 workaround for pyarrow loaded
import os
import json
from .arg_parser import preprocess_parser
from pytorch_pretrained_bert.tokenization import BertTokenizer
from .utils.utils import MTDNNSSTConstants
from azureml.studio.common.logger import module_logger, TimeProfile
from azureml.studio.common.datatable.data_table import DataTable
from azureml.studio.common.datatypes import DataTypes
from azureml.studio.modulehost.handler.port_io_handler import OutputHandler


class MTDNNSSTPreprocess:
    def __init__(self, meta: dict):
        self.do_lower_case = meta.get('Do lower case', False)
        self.max_seq_len = int(meta.get('Maximum sequence length', 512))
        self.model = str(meta.get('Based BERT model'))
        return

    def run(self, input_df: pd.DataFrame, meta: dict = None):
        with TimeProfile("Applying preprocess."):
            assert 1 <= len(input_df.columns) <= 2

            # load tokenizer
            tokenizer = BertTokenizer.from_pretrained(self.model, do_lower_case=self.do_lower_case)

            is_with_label = MTDNNSSTConstants.LabelColumn in input_df.columns

            # build data
            uid_list = []
            token_id_list = []
            label_list = []
            type_id_list = []
            premise_list = []
            for idx, row in input_df.iterrows():
                uid = idx
                premise = row[MTDNNSSTConstants.TextColumn]
                if len(premise) > self.max_seq_len - 2:
                    premise = premise[:self.max_seq_len - 2]
                input_ids, _, type_ids = MTDNNSSTPreprocess._bert_feature_extractor(premise,
                                                                                    max_seq_length=self.max_seq_len,
                                                                                    tokenize_fn=tokenizer)
                uid_list.append(uid)
                token_id_list.append(str(input_ids))
                if is_with_label:
                    label_list.append(int(row[MTDNNSSTConstants.LabelColumn]))
                type_id_list.append(str(type_ids))
                premise_list.append(premise)
            if is_with_label:
                output_df = pd.DataFrame(
                    {MTDNNSSTConstants.UidColumn: uid_list, MTDNNSSTConstants.TokenColumn: token_id_list,
                     MTDNNSSTConstants.LabelColumn: label_list, MTDNNSSTConstants.TypeIdColumn: type_id_list,
                     MTDNNSSTConstants.TextColumn: premise_list})
            else:
                output_df = pd.DataFrame(
                    {MTDNNSSTConstants.UidColumn: uid_list, MTDNNSSTConstants.TokenColumn: token_id_list,
                     MTDNNSSTConstants.TypeIdColumn: type_id_list, MTDNNSSTConstants.TextColumn: premise_list})
            output_dt = DataTable(output_df)

        return output_dt

    @staticmethod
    def _truncate_seq_pair(tokens_a, tokens_b, max_length):
        """Truncates a sequence pair in place to the maximum length.
        Copyed from https://github.com/huggingface/pytorch-pretrained-BERT
        """
        # This is a simple heuristic which will always truncate the longer sequence
        # one token at a time. This makes more sense than truncating an equal percent
        # of tokens from each, since if one sequence is very short then each token
        # that's truncated likely contains more information than a longer sequence.
        while True:
            total_length = len(tokens_a) + len(tokens_b)
            if total_length <= max_length:
                break
            if len(tokens_a) > len(tokens_b):
                tokens_a.pop()
            else:
                tokens_b.pop()

    @staticmethod
    def _bert_feature_extractor(
            text_a, text_b=None, max_seq_length=512, tokenize_fn=None):
        tokens_a = tokenize_fn.tokenize(text_a)
        tokens_b = None
        if text_b:
            tokens_b = tokenize_fn.tokenize(text_b)

        if tokens_b:
            MTDNNSSTPreprocess._truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
        else:
            # Account for one [SEP] & one [CLS] with "- 2"
            if len(tokens_a) > max_seq_length - 2:
                tokens_a = tokens_a[:max_seq_length - 2]
        if tokens_b:
            input_ids = tokenize_fn.convert_tokens_to_ids(
                ['[CLS]'] + tokens_b + ['[SEP]'] + tokens_a + ['[SEP]'])
            segment_ids = [0] * (len(tokens_b) + 2) + [1] * (len(tokens_a) + 1)
        else:
            input_ids = tokenize_fn.convert_tokens_to_ids(
                ['[CLS]'] + tokens_a + ['[SEP]'])
            segment_ids = [0] * len(input_ids)
        input_mask = None
        return input_ids, input_mask, segment_ids

    @staticmethod
    def read_parquet(input_dir):
        df = pd.read_parquet(os.path.join(input_dir, MTDNNSSTConstants.InputFile), engine='pyarrow')
        return df


def main():
    parser = preprocess_parser()
    args = parser.parse_args()

    meta = {'Based BERT model': args.model,
            'Do lower case': args.do_lower_case,
            'Maximum sequence length': args.max_seq_len}

    preprocessor = MTDNNSSTPreprocess(meta)
    module_logger.info("Loading dataset to apply mtdnn sst preprocess.")
    input_df = MTDNNSSTPreprocess.read_parquet(args.input_dir)
    output_dt = preprocessor.run(input_df)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    OutputHandler.handle_output(data=output_dt, file_path=args.output_dir, file_name=MTDNNSSTConstants.PreprocessedFile,
                                data_type=DataTypes.DATASET)

    # Dump data_type.json as a work around until SMT deploys
    dct = {
        "Id": "Dataset",
        "Name": "Dataset .NET file",
        "ShortName": "Dataset",
        "Description": "A serialized DataTable supporting partial reads and writes",
        "IsDirectory": False,
        "Owner": "Microsoft Corporation",
        "FileExtension": "dataset.parquet",
        "ContentType": "application/octet-stream",
        "AllowUpload": False,
        "AllowPromotion": True,
        "AllowModelPromotion": False,
        "AuxiliaryFileExtension": None,
        "AuxiliaryContentType": None
    }
    with open(os.path.join(args.output_dir, 'data_type.json'), 'w') as f:
        json.dump(dct, f)

    return


if __name__ == "__main__":
    main()
