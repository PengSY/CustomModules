import pandas as pd
import pyarrow.parquet as pq  # noqa: F401 workaround for pyarrow loaded
import json
import os
import torch
from .arg_parser import score_parser
from .mtdnn.batcher import BatchGen
from .mtdnn.model import MTDNNModel
from .utils.utils import eval_model, MTDNNSSTConstants, setup_logger
from azureml.studio.common.datatable.data_table import DataTable
from azureml.studio.common.datatypes import DataTypes
from azureml.studio.modulehost.handler.port_io_handler import OutputHandler
from .utils.metrics import Metric


class MTDNNSSTScore:
    def __init__(self, trained_model_dir, meta: dict):
        # load trained model
        self.logger = setup_logger(MTDNNSSTConstants.ScoreLogger, MTDNNSSTConstants.ScoreLogFile)
        opt = json.load(open(os.path.join(trained_model_dir, MTDNNSSTConstants.ModelMetaFile)))
        opt["batch_size"] = meta["Test batch size"]
        opt["cuda"] = meta["Use cuda"]

        if opt["cuda"] and not torch.cuda.is_available():
            self.logger.info("The compute doesn't have a NVIDIA GPU, changed to use CPU.")
            opt["cuda"] = False

        self.opt = opt
        self.model = MTDNNModel(opt)
        self.model.load(os.path.join(trained_model_dir, MTDNNSSTConstants.TrainedModel))

        return

    def load_parquet_data(self, test_data_dir):
        self.logger.info("Loading data.")
        test_data_path = os.path.join(test_data_dir, MTDNNSSTConstants.PreprocessedFile)
        # test_data = BatchGen.load_parquet(path=test_data_path, is_train=False, maxlen=self.opt["max_seq_len"])
        test_data = pd.read_parquet(test_data_path, engine='pyarrow')
        return test_data

    def run(self, test_data_df: pd.DataFrame, meta: dict = None):
        with_label = MTDNNSSTConstants.LabelColumn in test_data_df.columns
        test_data = BatchGen.load_dataframe(test_data_df)
        test_data = BatchGen(data=test_data,
                             batch_size=self.opt["batch_size"],
                             dropout_w=self.opt["dropout_w"],
                             gpu=self.opt["cuda"],
                             maxlen=self.opt["max_seq_len"],
                             is_train=False)
        sst_metric = tuple(Metric[metric_name] for metric_name in MTDNNSSTConstants.SSTMetric)
        metrics, pred, _, _, ids = eval_model(self.model, test_data, metric_meta=sst_metric,
                                              use_cuda=self.opt["cuda"],
                                              with_label=with_label)
        result_df = pd.DataFrame({MTDNNSSTConstants.UidColumn: ids, MTDNNSSTConstants.ScoreColumn: pred})
        test_data_df[MTDNNSSTConstants.TypeIdColumn] = test_data_df[MTDNNSSTConstants.TypeIdColumn].apply(
            lambda x: str(x))
        test_data_df[MTDNNSSTConstants.TokenColumn] = test_data_df[MTDNNSSTConstants.TokenColumn].apply(
            lambda x: str(x))
        result_df = test_data_df.join(result_df.set_index(MTDNNSSTConstants.UidColumn), on=MTDNNSSTConstants.UidColumn)
        result_dt = DataTable(result_df)
        if with_label:
            metric_str = str()
            for metric in MTDNNSSTConstants.SSTMetric:
                metric_str += f"{metric}: {metrics[metric]}\n"
            self.logger.info(f"Evaluate model: \n{metric_str}")
        return result_dt


def main():
    parser = score_parser()
    args = parser.parse_args()

    meta = {'Use cuda': args.cuda,
            'Test batch size': args.batch_size}
    mtdnn_sst_score = MTDNNSSTScore(args.trained_model_dir, meta)
    # load data
    test_data = mtdnn_sst_score.load_parquet_data(args.test_data_dir)
    # score
    result_dt = mtdnn_sst_score.run(test_data)
    # save
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    OutputHandler.handle_output(data=result_dt, file_path=args.output_dir, file_name=MTDNNSSTConstants.ScoreFile,
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
