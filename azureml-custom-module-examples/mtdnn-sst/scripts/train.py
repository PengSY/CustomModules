import pyarrow.parquet as pq  # noqa: F401 workaround for pyarrow loaded
import torch
import os
import json
from datetime import datetime
from .arg_parser import train_parser
from .mtdnn.batcher import BatchGen
from .mtdnn.model import MTDNNModel
from .utils.utils import MTDNNSSTConstants, setup_logger


def main():
    logger = setup_logger(MTDNNSSTConstants.TrainLogger, MTDNNSSTConstants.TrainLogFile)
    parser = train_parser()
    args = parser.parse_args()
    args.init_checkpoint = os.path.join(args.init_checkpoint_dir, MTDNNSSTConstants.InitCheckpointFile)

    if args.cuda and not torch.cuda.is_available():
        logger.info("The compute doesn't have a NVIDIA GPU, changed to use CPU.")
        args.cuda = False

    opt = vars(args)
    opt['answer_opt'] = [opt['answer_opt']]
    opt['tasks_dropout_p'] = [args.dropout_p]

    # load data
    logger.info("Loading training data.")
    train_data_path = os.path.join(args.train_data_dir, MTDNNSSTConstants.PreprocessedFile)
    train_data = BatchGen.load_parquet(path=train_data_path, is_train=True, maxlen=args.max_seq_len)
    train_data = BatchGen(data=train_data,
                          batch_size=args.batch_size,
                          dropout_w=args.dropout_w,
                          gpu=args.cuda,
                          maxlen=args.max_seq_len)
    train_iter = iter(train_data)
    # load model
    logger.info("Loading init model.")
    if args.cuda:
        state_dict = torch.load(args.init_checkpoint)
    else:
        state_dict = torch.load(args.init_checkpoint, map_location="cpu")
    config = state_dict['config']
    config['attention_probs_dropout_prob'] = args.bert_dropout_p
    config['hidden_dropout_prob'] = args.bert_dropout_p
    opt.update(config)

    # init model
    num_all_batches = args.epochs * len(train_data) // args.grad_accumulation_step
    model = MTDNNModel(opt, state_dict=state_dict, num_train_step=num_all_batches)

    # train model
    start_time = datetime.now()
    for epoch in range(0, args.epochs):
        train_data.reset()
        for i in range(len(train_data)):
            batch_meta, batch_data = next(train_iter)
            model.update(batch_meta, batch_data)
        remaining_time = str((datetime.now() - start_time) / (epoch + 1) * (args.epochs - epoch - 1)).split(".")[0]
        logger.info(
            f"Epoch[{epoch:2}] train loss[{model.train_loss.avg:.5f}] remaining[{remaining_time:3}]")

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    # save model
    logger.info("Saving MT-DNN model.")
    model_save_path = os.path.join(args.output_dir, MTDNNSSTConstants.TrainedModel)
    model.save(model_save_path)
    # save model meta
    json.dump(opt, open(os.path.join(args.output_dir, MTDNNSSTConstants.ModelMetaFile), "w"))

    # Dump data_type.json as a work around until SMT deploys
    dct = {
        "Id": "ILearnerDotNet",
        "Name": "ILearner .NET file",
        "ShortName": "Model",
        "Description": "A .NET serialized ILearner",
        "IsDirectory": False,
        "Owner": "Microsoft Corporation",
        "FileExtension": "ilearner",
        "ContentType": "application/octet-stream",
        "AllowUpload": False,
        "AllowPromotion": False,
        "AllowModelPromotion": True,
        "AuxiliaryFileExtension": None,
        "AuxiliaryContentType": None
    }
    with open(os.path.join(args.output_dir, 'data_type.json'), 'w') as f:
        json.dump(dct, f)
    # Dump data.ilearner as a work around until data type design
    visualization = os.path.join(args.output_dir, "data.ilearner")
    with open(visualization, 'w') as file:
        file.writelines('{}')

    return


if __name__ == "__main__":
    main()
