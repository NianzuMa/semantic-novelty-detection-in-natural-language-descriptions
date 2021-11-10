import os
import sys

currentdir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(currentdir)

currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

import shutil
import json
import torch
import random
import math
import numpy as np
from tqdm import tqdm
from transformers import (WEIGHTS_NAME, BertConfig,
                          BertForSequenceClassification, BertTokenizer,
                          RobertaConfig,
                          RobertaForSequenceClassification,
                          RobertaTokenizer,
                          XLMConfig, XLMForSequenceClassification,
                          XLMTokenizer, XLNetConfig,
                          XLNetForSequenceClassification,
                          XLNetTokenizer,
                          DistilBertConfig,
                          DistilBertForSequenceClassification,
                          DistilBertTokenizer)

MODEL_CLASSES = {
    'bert': (BertConfig, BertForSequenceClassification, BertTokenizer),
    'xlnet': (XLNetConfig, XLNetForSequenceClassification, XLNetTokenizer),
    'xlm': (XLMConfig, XLMForSequenceClassification, XLMTokenizer),
    'roberta': (RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer),
    'distilbert': (DistilBertConfig, DistilBertForSequenceClassification, DistilBertTokenizer)
}

from data_utils import GAT_Factual_Reasoning_Processor
from GAT_models import Net
import evaluation_metrics
import pytorch_utils
import time
from torch.utils.tensorboard import SummaryWriter


def argument_parser():
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument("--data_folder", default="", type=str,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")

    parser.add_argument("--lowest_train_loss_model", default="lowest_train_loss_model.pt", type=str,
                        help="")

    parser.add_argument("--collected_data_folder", default="",
                        type=str,
                        help="collect data of all verb and put together")

    parser.add_argument("--wordnet_synset_similarity_dict_folder", default="./wordnet_synset_similarity_dict_folder",
                        type=str,
                        help="wordnet_synset_similarity_dict_folder")
    parser.add_argument("--output_dir", default="", type=str,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--output_tag", type=str, default="hypernym_without_online_parsing",
                        help="")

    parser.add_argument('--logging_steps', type=int, default=1,
                        help="Log every X updates steps.")

    # feature creation method
    parser.add_argument("--additional_feature", default="hypernym_embed", type=str,
                        help="use hypernym feature for entities")
    # additional_feature choices:
    # ["hypernym_embed", "nothing"]

    parser.add_argument("--feature_type", default="bert", type=str, help="The method to create features")
    # feature creation choice:
    # ["glove",
    #  "bert"
    #  ]
    parser.add_argument("--glove_embedding_type", default="glove_6B_300d",
                        help="glove_6B_50d | "
                             "glove_6B_100d | "
                             "glove_6B_200d | "
                             "glove_6B_300d | "
                             "glove_42B |"
                             "glove_840B")
    parser.add_argument("--glove_embed_size", type=int, default=300, help="should consistent with above")
    parser.add_argument("--glove_embedding_folder", default="", help="glove data folder")
    parser.add_argument("--CORENLP_HOME", type=str, default="",
                        help="stanford corenlp parser. If use the same one for two instances, there will be conflict.")
    parser.add_argument("--corenlp_ports", type=int, default=9001)
    parser.add_argument("--pretrained_transformer_model_type", default="bert", type=str,
                        help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()))
    parser.add_argument("--pretrained_transformer_model_name_or_path", default="bert-base-uncased", type=str,
                        help="Path to pre-trained model or shortcut name selected in the list")
    parser.add_argument("--pretrained_bert_model_folder_for_feature_creation",
                        default=None,
                        type=str,
                        help="the pretrained model to create bert embedding for following task.")

    parser.add_argument("--input_size", default=768, type=int, help="initial input size")
    parser.add_argument("--hidden_size", default=300, type=int, help="the hidden size for GAT layer")

    parser.add_argument("--heads", default=6, type=int, help="number of heads in the GAT layer")
    parser.add_argument("--att_dropout", default=0, type=float, help="")
    parser.add_argument("--stack_layer_num", default=8, type=int, help="the number of layers to stack")

    parser.add_argument("--num_classes", default=-1, type=int, help="the number of class")
    parser.add_argument("--embed_dropout", default=0.3, type=float, help="dropout for input word embedding")

    parser.add_argument("--max_seq_length", default=100, type=int,
                        help="The maximum total input sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded. Acutally as long as the max"
                             "lenght is lower than BERT max length, it is OK. The max lenght is 86 for bert tokenizer. "
                             "so it is OK")

    parser.add_argument("--do_lower_case", action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--per_gpu_train_batch_size", default=128, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--per_gpu_eval_batch_size", default=32, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")

    parser.add_argument("--learning_rate", default=0.0005, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=1e-4, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--num_train_epochs", default=100, type=int,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--warmup_steps", default=0, type=int,
                        help="Linear warmup over warmup_steps.")
    parser.add_argument('--save_steps', type=int, default=50,
                        help="Save checkpoint every X updates steps.")
    parser.add_argument("--eval_all_checkpoints", action='store_true',
                        help="Evaluate all checkpoints starting with the same prefix as model_name ending and ending with step number")

    parser.add_argument('--overwrite_output_dir', action='store_true',
                        help="Overwrite the content of the output directory")
    parser.add_argument('--overwrite_cache', action='store_true',
                        help="Overwrite the cached training and evaluation sets")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument('--fp16', action='store_true',
                        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit")
    parser.add_argument('--fp16_opt_level', type=str, default='O1',
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See details at https://nvidia.github.io/apex/amp.html")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="For distributed training: local_rank")
    parser.add_argument('--server_ip', type=str, default='', help="For distant debugging.")
    parser.add_argument('--server_port', type=str, default='', help="For distant debugging.")

    parser.add_argument("--best_dev_eval_file", type=str, default="best_dev_info.json")
    parser.add_argument("--considered_metrics", type=str, default="auc_score")
    parser.add_argument("--save_model_file_name", type=str, default="best_model.pt")
    parser.add_argument("--output_mode", type=str, default="classification")

    parser.add_argument("--do_train", action='store_true', default=True,
                        help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true', default=True,
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--evaluate_during_training", action='store_true', default=True,
                        help="Rul evaluation during training at each logging step.")

    parser.add_argument("--do_test", action="store_true", default=True,
                        help="load the trained model and check testing result")
    parser.add_argument("--trained_result_folder", type=str, default="./result/2020-04-21__09-49__543108")

    args = parser.parse_args()

    return args


def train_dynamic_sampled_GAT(args, model_class, tokenizer_class, model, processor):
    """ Train the model
    """
    train_dataset = processor.get_train_examples()
    total_label_list = processor.get_labels()

    # set up tensorboard writer
    tb_writer_train = SummaryWriter(os.path.join(args.output_dir, "tb/train"))
    tb_writer_valid = SummaryWriter(os.path.join(args.output_dir, "tb/valid"))
    tb_writer_test = SummaryWriter(os.path.join(args.output_dir, "tb/test"))

    # set optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    batch_size = args.per_gpu_train_batch_size
    total_train_size = len(train_dataset)
    batch_num = math.ceil(total_train_size * 1.0 / batch_size)

    print("***** Running training *****")
    print(f"  Num examples = {len(train_dataset)}")
    print(f"  Num Epochs = {args.num_train_epochs}")
    print(f"  Num Batch_size = {batch_size}")
    print(f"  Instantaneous batch size per GPU = {args.per_gpu_train_batch_size}")
    print(f"  Num of steps per epoch = {batch_num}")
    print(f"  Total optimization steps = {args.num_train_epochs * batch_num}")

    pytorch_utils.set_seed(args)

    # train_dataset is a feature list
    global_step = 0
    total_train_loss = 0
    best_eval_score = 0
    lowest_train_loss = sys.float_info.max

    all_synset_set = processor.get_entity_in_training_data_sampling_negative_example_dictionary(train_dataset)
    print(f"There are totally {len(all_synset_set)} synset in the training data")

    for epoch_index in range(args.num_train_epochs):
        epoch_begin_time = time.time()

        # shuffle
        random.shuffle(train_dataset)
        print("The train dataset is shuffling for epoch {}".format(epoch_index))

        for batch_index in range(batch_num):
            global_step += 1

            print(
                f"\n\n\n>>>>>>>>>>>>>> batch[{batch_index}/{batch_num}] -- epoch[{epoch_index}] -- global_step[{global_step}] <<<<<<<<<<<<<<<<<<<<<<")
            batch_begin_time = time.time()

            model.train()
            model.zero_grad()

            cur_batch_examples = train_dataset[batch_index * batch_size: (batch_index + 1) * batch_size]

            total_model_input = processor.dynamically_get_positive_and_negative_batch_without_online_parsing(args,
                                                                                                             cur_batch_examples)

            word_embed_matrix, graph_edge_list, target_mask_list, label_id_list, input_token_size_list = total_model_input

            # load on device
            word_embed_matrix = torch.tensor(word_embed_matrix, dtype=torch.float).to(args.device)
            graph_edge_list = torch.tensor(np.array(graph_edge_list), dtype=torch.long).t().contiguous().to(
                args.device)
            label_id_list = torch.tensor(label_id_list, dtype=torch.long).to(args.device)

            total_score, loss = model(args, word_embed_matrix, target_mask_list, graph_edge_list, label_id_list)
            loss.backward()
            optimizer.step()
            model.zero_grad()

            total_train_loss += loss.item()

            cur_train_loss = total_train_loss * 1.0 / global_step

            train_info_json = {"epoch": epoch_index, "batch": f"{batch_index}/{batch_num}", "global_step": global_step,
                               "train_loss": cur_train_loss}
            print(f"{'#' * 30} TRAIN: {str(train_info_json)} {'#' * 30}")

            lowest_train_loss_model_path = os.path.join(args.output_dir, "best_model", args.lowest_train_loss_model)
            if cur_train_loss < lowest_train_loss:
                lowest_train_loss = cur_train_loss
                subdir = os.path.join(args.output_dir, "best_model")
                if not os.path.exists(subdir):
                    os.makedirs(subdir)
                # endif
                pytorch_utils.save_model(model, lowest_train_loss_model_path)
            else:
                model = pytorch_utils.load_model(model, args.device, lowest_train_loss_model_path)
                with open(os.path.join(args.output_dir, "train_error_results.txt"), mode="a") as fout:
                    fout.write(json.dumps(train_info_json) + "\n")
                # endwith
                continue
            # endif

            tb_writer_train.add_scalar('Loss', cur_train_loss, global_step)

            with open(os.path.join(args.output_dir, "train_results.txt"), mode="a") as fout:
                fout.write(json.dumps(train_info_json) + "\n")
            # endwith

            if global_step % args.logging_steps == 0:
                if args.evaluate_during_training:
                    valid_results_json_dict = evaluate(args, processor, model_class, tokenizer_class, model,
                                                       mode="valid", epoch_index=epoch_index, step=global_step)
                    print(f"{'#' * 30} VALID: {str(valid_results_json_dict)} {'#' * 30}")
                    valid_auc_score = valid_results_json_dict["auc_score"]
                    tb_writer_valid.add_scalar("valid_auc", valid_auc_score, global_step)

                    if_best_model = False
                    test_results_json_dict = None
                    # save model if the model give the best metrics we care
                    current_eval_score = valid_results_json_dict[args.considered_metrics]
                    if current_eval_score > best_eval_score:
                        if_best_model = True
                        # save the best model
                        best_eval_score = current_eval_score
                        subdir = os.path.join(args.output_dir, "best_model")
                        if not os.path.exists(subdir):
                            os.makedirs(subdir)
                        # endif
                        pytorch_utils.save_model(model,
                                                 os.path.join(args.output_dir, "best_model", args.save_model_file_name))

                        # save the best eval information
                        with open(os.path.join(args.output_dir, "best_valid_result.json"), mode="w") as fout:
                            fout.write(json.dumps(valid_results_json_dict) + "\n")
                        # endwith

                        # use the current best model to evaluate on test data
                        test_results_json_dict = evaluate(args, processor, model_class, tokenizer_class, model,
                                                          mode="test", if_write_pred_result=True,
                                                          epoch_index=epoch_index, step=global_step)
                        print(f"{'#' * 30} TEST: {test_results_json_dict} {'#' * 30}")

                        # save the best test information, and history information
                        with open(os.path.join(args.output_dir, "best_test_result.json"), mode="w") as fout:
                            fout.write(json.dumps(test_results_json_dict) + "\n")
                        # endwith

                        with open(os.path.join(args.output_dir, "best_valid_test_result_log.json"), mode="a") as fout:
                            fout.write("valid: " + json.dumps(valid_results_json_dict) + "\n")
                            fout.write("test:  " + json.dumps(test_results_json_dict) + "\n")
                            fout.write("\n")
                        # endwith
                    # endif

                    if not if_best_model:
                        test_results_json_dict = evaluate(args, processor, model_class, tokenizer_class, model,
                                                          mode="test", if_write_pred_result=True,
                                                          epoch_index=epoch_index, step=global_step)
                        print(f"{'#' * 30} TEST: {test_results_json_dict} {'#' * 30}")
                    # endif
                    assert test_results_json_dict is not None
                    test_auc_score = test_results_json_dict["auc_score"]
                    tb_writer_test.add_scalar("test_auc", test_auc_score, global_step)

                # endif
            # endif

            # end of batch
            batch_time_length_min = (time.time() - batch_begin_time) * 1.0 / 60
            print(f">>>>>>>>>>>>>>> This batch takes {batch_time_length_min} min")
            batch_time_json = {"batch_index": batch_index, "time_mins": batch_time_length_min}
            with open(os.path.join(args.output_dir, "batch_time_log.txt"), mode="a") as fout:
                fout.write(f"{json.dumps(batch_time_json)}\n")
            # endwith
        # endfor

        # end of epoch
        epoch_time_length_min = (time.time() - epoch_begin_time) * 1.0 / 60
        print(f">>>>>>>>>>>>>>>>>> This epoch takes {epoch_time_length_min} mins")
        epoch_time_json = {"epoch_index": epoch_index, "time_mins": epoch_time_length_min}
        with open(os.path.join(args.output_dir, "epoch_time_log.txt"), mode="a") as fout:
            fout.write(f"{json.dumps(epoch_time_json)}\n")
        # endwith


def evaluate(args, processor, model_class, tokenizer_class, model, mode=None,
             if_write_pred_result=False,
             epoch_index=None,
             step=None,
             output_file=None):
    assert epoch_index is not None
    assert mode in ["train", "valid", "test"]

    eval_output_dir = args.output_dir

    results = {}
    eval_dataset = load_and_cache_examples(args,
                                           processor,
                                           model_class,
                                           tokenizer_class,
                                           mode=mode)

    if not os.path.exists(eval_output_dir) and args.local_rank in [-1, 0]:
        os.makedirs(eval_output_dir)

    eval_batch_size = args.per_gpu_eval_batch_size
    total_eval_size = len(eval_dataset)
    batch_num = math.ceil(total_eval_size * 1.0 / eval_batch_size)

    # Eval!
    print(f"***** Running evaluation for {mode} dataset*****")
    print(f"  Num examples = {len(eval_dataset)}")
    print(f"  Batch size = {args.per_gpu_eval_batch_size}")
    print(f"  Batch num per epoch = {batch_num}")

    nb_eval_steps = 0
    preds = None
    out_label_ids = None

    for batch_index in tqdm(range(batch_num), desc="Evaluating"):
        model.eval()

        word_embed_matrix, graph_edge_list, target_mask_list, label_id_list, input_token_size_list = \
            processor.get_model_input(eval_dataset[batch_index * eval_batch_size: (batch_index + 1) * eval_batch_size])

        word_embed_matrix = torch.tensor(word_embed_matrix, dtype=torch.float).to(args.device)
        graph_edge_list = torch.tensor(np.array(graph_edge_list).transpose(), dtype=torch.long).to(args.device)
        label_id_list = torch.tensor(label_id_list, dtype=torch.long).to(args.device)

        with torch.no_grad():
            total_score, loss = model(args,
                                      word_embed_matrix,
                                      target_mask_list,
                                      graph_edge_list,
                                      label_id_list=None)
        # endwith

        nb_eval_steps += 1
        if preds is None:
            preds = total_score.detach().cpu().numpy()
            out_label_ids = label_id_list.detach().cpu().numpy()
        else:
            preds = np.append(preds, total_score.detach().cpu().numpy(), axis=0)
            out_label_ids = np.append(out_label_ids, label_id_list.detach().cpu().numpy(), axis=0)
        # endif
    # endfor

    auc_score = evaluation_metrics.get_auc_score(out_label_ids, preds)  # roc_auc_score(y_true, pred_score)
    results["auc_score"] = auc_score

    if output_file is None:
        output_eval_file = os.path.join(eval_output_dir, mode + "_results.txt")
    else:
        output_eval_file = output_file
    # endif

    with open(output_eval_file, "a") as writer:
        output_json = {'epoch': epoch_index,
                       'step': step,
                       'mode': mode}
        output_json.update(results)
        writer.write(json.dumps(output_json) + '\n')
    # endwith

    return output_json


def load_and_cache_examples(args,
                            processor,
                            model_class,
                            tokenizer_class,
                            mode=None):
    assert mode in ["valid", "test"]
    assert processor is not None
    feature_type = args.feature_type

    if feature_type == "glove":
        assert model_class == None
        assert tokenizer_class == None
    elif "bert" in feature_type:
        assert model_class is not None
        assert tokenizer_class is not None
    # endif

    cached_features_file = None
    if args.feature_type == "bert":
        cached_features_file = os.path.join(args.collected_data_folder,
                                            f"{mode}_cached_feature_{args.max_seq_length}_{args.feature_type}_{args.additional_feature}")
    if args.feature_type == "glove":
        cached_features_file = os.path.join(args.collected_data_folder,
                                            f"{mode}_cached_feature_{args.max_seq_length}_{args.feature_type}_{args.glove_embedding_type}_{args.additional_feature}")

    assert cached_features_file is not None, "{} is None".format(cached_features_file)

    # if cased file exists, load file
    if os.path.exists(cached_features_file):
        print(f"Loading features from cached file {cached_features_file}")
        features = torch.load(cached_features_file, map_location=args.device.type)
        print(f"dataset is loaded to device: {args.device.type}")
    else:
        if processor.glove_embed_handler is None:
            if args.feature_type == "glove":
                print("loading gensim model ...")
                processor._load_gensim_model()
                print("gensim model: {} is loaded".format(args.glove_embedding_type))
            # endif

        print(f"Creating features from dataset file at {args.data_folder}")
        label_list = processor.get_labels()

        examples = None
        if mode == "valid":
            examples = processor.get_valid_examples()
        if mode == "test":
            examples = processor.get_test_examples()
        # endif
        assert examples is not None

        features = None
        if args.additional_feature == "nothing":
            features = processor.convert_examples_to_features(examples=examples, mode=mode)

        if args.additional_feature == "hypernym_embed":
            features = processor.convert_examples_to_features_with_hypernym(examples=examples, mode=mode)

        assert features is not None
        print("Saving features into cached file {}".format(cached_features_file))
        torch.save(features, cached_features_file)
    # endif

    return features


def save_model(model, tokenizer, args):
    model_to_save = model.module if hasattr(model,
                                            'module') else model  # Take care of distributed/parallel training
    model_to_save.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    pass


def load_model(model_class, tokenizer_class, args):
    model = model_class.from_pretrained(args.output_dir)
    tokenizer = tokenizer_class.from_pretrained(args.output_dir)
    model.to(args.device)
    return model, tokenizer


def load_GAT_model(model, path):
    model.load_state_dict(torch.load(path))
    return model
    pass


def _validate_input_size(args):
    if args.feature_type == "glove":
        assert args.glove_embed_size == args.input_size == 300, f"args.glove_embed_size: {args.glove_embed_size} - args.input_size: {args.input_size}"

    if "bert" in args.feature_type:
        if "bert-base" in args.pretrained_transformer_model_name_or_path:
            assert args.input_size == 768
        # endif
        if "bert-large" in args.pretrained_transformer_model_name_or_path:
            assert args.input_size == 1024
        # endif


def run_app():
    args = argument_parser()

    _validate_input_size(args)

    # create unique output folder based on time
    pytorch_utils.set_output_folder(args)
    print(f"output folder is {args.output_dir}")

    # Setup CUDA, GPU
    pytorch_utils.set_device(args)
    print(f"Device: {args.device}")

    # Set seed
    pytorch_utils.set_seed(args)

    # write parameter
    pytorch_utils.write_params(args)

    # processor
    processor = GAT_Factual_Reasoning_Processor(args)
    assert processor is not None

    total_label_list = processor.get_labels()
    args.num_classes = len(total_label_list)

    if args.feature_type == "glove":
        config_class, model_class, tokenizer_class = None, None, None
    else:
        config_class, model_class, tokenizer_class = MODEL_CLASSES[args.pretrained_transformer_model_type]
    # endif

    model = Net(args)
    model.to(args.device)

    # Training
    if args.do_train:
        train_dynamic_sampled_GAT(args, model_class, tokenizer_class, model, processor)
    # endif
    try:
        move_log_file_to_output_directory(args.output_dir)
    except:
        print("cannot find log.txt file")
    # endfor


def _delete_cached_feature_file(args):
    for dir, subdir, file_list in os.walk(args.data_folder):
        for file in file_list:
            if "cached" in file:
                file_path = os.path.join(dir, file)
                os.remove(file_path)
            # endif
        # endfor
    # endfor


def move_log_file_to_output_directory(output_dir):
    shutil.move("./log.txt", output_dir)


if __name__ == '__main__':
    run_app()
