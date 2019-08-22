import json
import os
import pandas as pd
from .arg_parser import score_parser
from .mtdnn.batcher import BatchGen
from .mtdnn.model import MTDNNModel
from .utils.utils import eval_model, MTDNNSSTConstants
from azureml.studio.common.logger import module_logger,TimeProfile


class MTDNNSSTScore:
    def __init__(self, trained_model_dir, meta: dict):
        # load trained model
        opt = json.load(open(os.path.join(trained_model_dir, MTDNNSSTConstants.ModelMetaFile)))
        opt["batch_size"] = meta["Test batch size"]
        opt["cuda"] = meta["Use cuda"]
        self.opt = opt
        self.model = MTDNNModel(opt)
        self.model.load(os.path.join(trained_model_dir, MTDNNSSTConstants.TrainedModel))

        return

    def load_parquet_data(self, test_data_dir):
        module_logger.info("Loading data.")
        test_data_path = os.path.join(test_data_dir, MTDNNSSTConstants.PreprocessedFile)
        test_data = BatchGen.load_parquet(path=test_data_path, is_train=False, maxlen=self.opt["max_seq_len"])
        return test_data

    def run(self, test_data: pd.DataFrame, meta: dict = None):
        test_data = BatchGen(data=test_data,
                             batch_size=self.opt["batch_size"],
                             dropout_w=self.opt["dropout_w"],
                             gpu=self.opt["cuda"],
                             maxlen=self.opt["max_seq_len"],
                             is_train=False)
        with TimeProfile("Evaluating model."):
            _, pred, _, _, ids = eval_model(self.model, test_data, metric_meta=MTDNNSSTConstants.SSTMetric,
                                            use_cuda=self.opt["cuda"],
                                            with_label=False)
        result_df = pd.DataFrame({MTDNNSSTConstants.IdColumn: ids, MTDNNSSTConstants.ScoreColumn: pred})
        return result_df


def main():
    parser = score_parser()
    args = parser.parse_args()

    meta = {'Use cuda': args.cuda,
            'Test batch size': args.batch_size}
    mtdnn_sst_score = MTDNNSSTScore(args.trained_model_dir, meta)
    # load data
    test_data = mtdnn_sst_score.load_parquet_data(args.test_data_dir)
    # score
    result_df = mtdnn_sst_score.run(test_data)
    # save
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    scored_file_save_path = os.path.join(args.output_dir, MTDNNSSTConstants.ScoreFile)
    result_df.to_parquet(scored_file_save_path, engine="pyarrow")

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
