from __future__ import absolute_import, division, print_function

from nltk.corpus import wordnet as wn
import copy
import torch
import time
import random

random.seed(1133)
import os
from tqdm import tqdm
from io import open
import json
import math
import numpy as np
from collections import Counter, defaultdict
from wordnet_utils import WordNetUtils
from multiprocessing import Manager
import multiprocessing


class GAT_Factual_Reasoning_Example():
    def __init__(self,
                 instance_id,
                 sent_text,
                 target_verb_lemma,
                 target_verb_index,
                 index_to_word_dict,
                 index_to_lemma_dict,
                 index_to_pos_dict,
                 directed_graph_edges,
                 undirected_graph_edges,
                 directed_dep_relation_info,
                 label,
                 entity_wordnet_info_list=None,
                 index_to_hypernym_list=None
                 ):
        self.instance_id = instance_id
        self.sent_text = sent_text
        self.target_verb_lemma = target_verb_lemma
        self.target_verb_index = target_verb_index
        self.index_to_word_dict = index_to_word_dict
        self.index_to_lemma_dict = index_to_lemma_dict
        self.index_to_pos_dict = index_to_pos_dict
        self.directed_graph_edges = directed_graph_edges
        self.undirected_graph_edges = undirected_graph_edges
        self.directed_dep_relation_info = directed_dep_relation_info
        self.label = label
        self.entity_wordnet_info_list = entity_wordnet_info_list
        self.index_to_hypernym_list = index_to_hypernym_list


class GAT_Factual_Reasoning_Feature():
    def __init__(self,
                 instance_id,
                 target_verb_index,
                 index_to_word_dict,
                 index_to_lemma_dict,
                 index_to_pos_dict,
                 word_embed_matrix,
                 directed_graph_edges,
                 undirected_graph_edges,
                 directed_dep_relation_info,
                 label_id):
        self.instance_id = instance_id
        self.target_verb_index = target_verb_index
        self.index_to_word_dict = index_to_word_dict
        self.index_to_lemma_dict = index_to_lemma_dict
        self.index_to_pos_dict = index_to_pos_dict
        self.word_embed_matrix = word_embed_matrix
        self.directed_graph_edges = directed_graph_edges
        self.undirected_graph_edges = undirected_graph_edges
        self.directed_dep_relation_info = directed_dep_relation_info
        self.label_id = label_id


class GAT_Factual_Reasoning_Processor():
    def __init__(self, args):
        self.args = args
        self.glove_embed_handler = None
        self.sampling_knowledge_dict = None
        self.bert_model = None
        self.bert_tokenizer = None
        self.corenlp_client = None
        self.sampling_knowledge_dict = {}
        self.sampling_freq_counter = defaultdict(int)

        # create directory
        wordnet_res_folder = os.path.join(args.output_dir, "wordnet_res")
        if not os.path.exists(wordnet_res_folder):
            os.makedirs(wordnet_res_folder)
        # endif

        self.wordnet_word_size_dict = self.build_wordnet_noun_word_size_dict()
        self.hypernym_error_set = set()

        self.load_omit_general_hypernym_set()
        assert len(self.omit_general_hypernym_set) > 0

        self.glove_embed_handler = None
        if args.feature_type == "glove" and self.glove_embed_handler is None:
            print("loading gensim model ...")
            self._load_gensim_model()
            print("gensim model: {} is loaded".format(args.glove_embedding_type))
        # endif
        pass

    def build_wordnet_noun_word_size_dict(self):
        wordnet_word_size_dict = defaultdict(set)

        all_noun_synset_list = list(wn.all_synsets('n'))
        print(f"There are totally {len(all_noun_synset_list)} nouns in the wordnet version 3.1")
        for tmp_synset in all_noun_synset_list:
            tmp_synset_name = tmp_synset._name
            tmp_str = tmp_synset_name.split(".")[0]
            tmp_str = tmp_str.replace("-", " ")
            tmp_str = tmp_str.replace("_", " ")
            tmp_word_list = tmp_str.split()
            tmp_word_length = len(tmp_word_list)
            wordnet_word_size_dict[tmp_word_length].add(tmp_synset_name)
        # endfor
        return wordnet_word_size_dict

    def load_omit_general_hypernym_set(self):
        omit_general_hypernym_path = "./wordnet_res/omit_general_hypernym.txt"
        assert os.path.exists(omit_general_hypernym_path)
        self.omit_general_hypernym_set = set()
        with open(omit_general_hypernym_path, mode="r") as fin:
            for line in fin:
                line = line.strip()
                if len(line) == 0:
                    continue
                # endif
                self.omit_general_hypernym_set.add(line)
            # endfor
        # endwith
        print(f"There are {len(self.omit_general_hypernym_set)} general hypernym item in set.")

    def _load_gensim_model(self):
        from word_embed import GloveEmbed
        glove_embed_handler = GloveEmbed(self.args.glove_embedding_folder, self.args.glove_embedding_type)
        glove_embed_handler.load_model(self.args.glove_embedding_type)
        self.glove_embed_handler = glove_embed_handler

    def _load_file_to_json_list(self, file_path):
        json_obj_list = []
        with open(file_path, mode="r") as fin:
            for line in fin:
                line = line.strip()
                if len(line) == 0:
                    continue
                # endif
                json_obj = json.loads(line)
                json_obj_list.append(json_obj)
            # endfor
        # endwith
        return json_obj_list

    def get_labels(self):
        return ["novel", "normal"]

    def _label_distribution(self, examples):
        print(f"dataset size: {len(examples)}")
        label_list = []
        for exp in examples:
            label_list.append(exp.label)
        # endfor
        label_dict = Counter(label_list)

        keys = label_dict.keys()
        keys = sorted(keys)
        print("=" * 8 + "label distribution" + "=" * 8)
        for k in keys:
            print("{}: {}".format(k, label_dict[k]), end=" ")
        # endfor
        print("")

    def _create_example_from_json_obj_list(self, json_obj_list, file_type):
        assert file_type in {"train", "valid", "test"}

        examples = []
        instance_id = 0
        for json_obj in json_obj_list:
            verb_lemma_in_relation_list = json_obj["target_verb_predicate_info"]

            assert verb_lemma_in_relation_list is not None
            target_verb_lemma, target_verb_index = verb_lemma_in_relation_list
            assert target_verb_lemma is not None and target_verb_index is not None
            # endif

            instance_id += 1
            uid = f"{file_type}_{str(instance_id)}"

            index_to_word_pos_dict = json_obj["index_to_word_pos_dict"]
            directed_dep_relation_info = json_obj["directed_dep_relation_info"]
            directed_graph_edges = json_obj["directed_graph_edges"]
            undirected_graph_edges = json_obj["undirected_graph_edges"]

            label = None
            if file_type == "train":
                # everything in training data is normal
                if "label" not in json_obj:
                    label = "normal"
                else:
                    label = json_obj["label"]
                    assert label == "novel"  # the negative example
            # endif

            if file_type in {"valid", "test"}:
                label = json_obj["label"]
            # endif

            assert label is not None

            index_to_word_dict = {}
            index_to_lemma_dict = {}
            index_to_pos_dict = {}
            word_list = []
            for index, info in index_to_word_pos_dict.items():
                index = int(index)
                word, lemma, pos = info
                index_to_word_dict[index] = word
                index_to_lemma_dict[index] = lemma
                index_to_pos_dict[index] = pos
                word_list.append(word)
            # endfor

            sent_text = " ".join(word_list)

            entity_wordnet_info_list = None
            if "entity_wordnet_info_list" in json_obj:
                entity_wordnet_info_list = json_obj["entity_wordnet_info_list"]

            index_to_hypernym_list = None
            if "index_to_hypernym_list" in json_obj:
                index_to_hypernym_list = json_obj["index_to_hypernym_list"]

            example = GAT_Factual_Reasoning_Example(instance_id=uid,
                                                    sent_text=sent_text,
                                                    target_verb_lemma=target_verb_lemma,
                                                    target_verb_index=target_verb_index,
                                                    index_to_word_dict=index_to_word_dict,
                                                    index_to_lemma_dict=index_to_lemma_dict,
                                                    index_to_pos_dict=index_to_pos_dict,
                                                    directed_graph_edges=directed_graph_edges,
                                                    undirected_graph_edges=undirected_graph_edges,
                                                    directed_dep_relation_info=directed_dep_relation_info,
                                                    label=label,
                                                    entity_wordnet_info_list=entity_wordnet_info_list,
                                                    index_to_hypernym_list=index_to_hypernym_list)

            examples.append(example)
        # endfor
        return examples

    def get_hypernym_list(self, synset_name):
        try:
            synset_item = wn.synset(synset_name)
            hypernym_list = list(set([s._name for s in synset_item.closure(lambda s: s.hypernyms())]))
        except:
            hypernym_list = None
            self.hypernym_error_set.add(synset_name)
        # endtry

        return hypernym_list

    def _get_entity_hypernym_information(self, json_obj):
        entity_wordnet_info_list = json_obj["entity_wordnet_info_list"]

        index_to_hypernym_list = []

        for index_list, synset_name in entity_wordnet_info_list:
            hypernym_list = self.get_hypernym_list(synset_name)
            if hypernym_list is not None:
                index_to_hypernym_list.append([index_list, hypernym_list])
        # endfor

        if len(index_to_hypernym_list) == 0:
            index_to_hypernym_list = None
        # endif

        json_obj["index_to_hypernym_list"] = index_to_hypernym_list
        return json_obj

    def compute_synset_ranking_list_multiprocess(self, to_compute_all_synset_str_set):
        manager = Manager()
        workers = []
        sampling_knowledge_dict = manager.dict()
        num_of_worker = 10
        all_synset_str_list = list(to_compute_all_synset_str_set)

        total_synset_num = len(all_synset_str_list)
        block_size = math.ceil(total_synset_num / num_of_worker)
        print(
            f"The block size is {block_size} of {num_of_worker} workers, totally {total_synset_num} synset to compute")

        for w in range(num_of_worker):
            block_list = all_synset_str_list[w * block_size: (w + 1) * block_size]
            workers.append(
                multiprocessing.Process(target=WordNetUtils.find_all_synset_similarity_ranking_list_multiprocess,
                                        args=(self.args.wordnet_synset_similarity_dict_folder,
                                              block_list)))
        # endfor

        for w in workers:
            w.start()
        # endfor

        for w in workers:
            w.join()
        # endfor
        pass

    def compute_synset_ranking_list_singleprocess(self, to_compute_all_synset_str_set):
        for tmp_synset in tqdm(to_compute_all_synset_str_set, desc="start computing synset knowledge single process"):
            WordNetUtils.find_all_synset_similarity_ranking_list_single_synset(
                self.args.wordnet_synset_similarity_dict_folder, tmp_synset)
        # endfor

    def get_entity_in_training_data_sampling_negative_example_dictionary(self, train_dataset):
        print(f"\n================================>>>> calculate wordnet entity knowledge ...")
        print(f"dump to folder {self.args.wordnet_synset_similarity_dict_folder}")
        begin = time.time()

        all_synset_str_set = set()

        for exp in train_dataset:
            entity_wordnet_info_list = exp.entity_wordnet_info_list

            for entity_info in entity_wordnet_info_list:
                entity_index_list, synset_str = entity_info
                assert synset_str is not None
                all_synset_str_set.add(synset_str)
            # endfor
        # endfor

        to_compute_all_synset_str_set = set()
        for tmp_synset in all_synset_str_set:
            synset_path = os.path.join(self.args.wordnet_synset_similarity_dict_folder, f"{tmp_synset}_sim.txt")
            if not os.path.exists(synset_path):
                to_compute_all_synset_str_set.add(tmp_synset)
            # endif
        # endfor

        print(f"There are {len(to_compute_all_synset_str_set)} to be computed.")
        print(f"{list(to_compute_all_synset_str_set)[:50]} ... ")

        # create folder
        if not os.path.exists(self.args.wordnet_synset_similarity_dict_folder):
            os.makedirs(self.args.wordnet_synset_similarity_dict_folder)
        # endif

        if len(to_compute_all_synset_str_set) > 10:
            self.compute_synset_ranking_list_multiprocess(to_compute_all_synset_str_set)
        else:
            self.compute_synset_ranking_list_singleprocess(to_compute_all_synset_str_set)
        # endif

        time_length = time.time() - begin
        print(f"Total time takes {time_length * 1.0 / 60} mins.")

        return all_synset_str_set

    def get_train_examples(self):
        train_data_file_path = os.path.join(self.args.collected_data_folder, "train_postprocessed.json")
        train_json_list = self._load_file_to_json_list(train_data_file_path)
        print(">" * 8 + f"train data instances in file: {len(train_json_list)}")

        examples = self._create_example_from_json_obj_list(train_json_list, "train")
        print(">" * 8 + f"train data examples correctly loaded: {len(examples)}")

        self._label_distribution(examples)
        return examples

    def get_valid_examples(self):
        valid_data_file_path = os.path.join(self.args.collected_data_folder, "valid_postprocessed.json")
        valid_json_list = self._load_file_to_json_list(valid_data_file_path)
        print(">" * 8 + f"valid data instance in file: {len(valid_json_list)}")

        examples = self._create_example_from_json_obj_list(valid_json_list, "valid")
        print(">" * 8 + f"valid data examples correctly loaded: {len(examples)}")
        self._label_distribution(examples)
        return examples

    def get_test_examples(self):
        test_data_file_path = os.path.join(self.args.collected_data_folder, "test_postprocessed.json")
        test_json_list = self._load_file_to_json_list(test_data_file_path)
        print(">" * 8 + f"test data instance loaded: {len(test_json_list)}")

        examples = self._create_example_from_json_obj_list(test_json_list, "test")
        print(">" * 8 + f"test data example correctly loaded: {len(examples)}")

        self._label_distribution(examples)
        return examples

    def get_bert_embed(self, examples, args):
        if self.bert_model is None and self.bert_tokenizer is None:
            from transformers import BertModel, BertTokenizer
            print("loading bert model: {} to create feature for sent tokens".format(
                args.pretrained_transformer_model_name_or_path))
            self.bert_model = BertModel.from_pretrained(args.pretrained_transformer_model_name_or_path)
            self.bert_tokenizer = BertTokenizer.from_pretrained(args.pretrained_transformer_model_name_or_path)
            self.bert_model.eval()
            self.bert_model.to(args.device)
            print(f"{args.pretrained_transformer_model_name_or_path} is successfully loaded.")

        bert_embeds_for_text_tokens_dict = {}
        for (ex_index, example) in enumerate(
                tqdm(examples, desc=f"create embedding from model - {args.pretrained_transformer_model_name_or_path}")):
            instance_id = example.instance_id

            index_to_word_dict = example.index_to_word_dict
            token_list = [word for index, word in index_to_word_dict.items()]

            word_pieces_list = []
            word_boundaries_list = []

            for w in token_list:
                word_pieces = self.bert_tokenizer.tokenize(w)
                word_boundaries_list.append([len(word_pieces_list), len(word_pieces_list) + len(
                    word_pieces)])
                word_pieces_list += word_pieces
            # endfor
            assert len(word_boundaries_list) == len(token_list)

            total_input_tokens = ['[CLS]'] + word_pieces_list + ['[SEP]']
            input_ids = self.bert_tokenizer.convert_tokens_to_ids(total_input_tokens)
            segment_ids = [0] * len(total_input_tokens)

            input_ids = torch.tensor([input_ids], dtype=torch.long).to(args.device)
            segment_ids = torch.tensor([segment_ids], dtype=torch.long).to(args.device)

            with torch.no_grad():
                sequence_output, _ = self.bert_model(input_ids=input_ids,
                                                     token_type_ids=segment_ids)
            # endwith

            sequence_output = sequence_output.squeeze(dim=0)
            text_piece_embeds = sequence_output[1:-1].to('cpu').numpy()  # [CLS] and [SEP] should be excluded

            text_token_embeds = []
            for i, w in enumerate(word_boundaries_list):
                text_token_embeds.append(
                    text_piece_embeds[word_boundaries_list[i][0]: word_boundaries_list[i][1]].mean(0))
            # endfor

            all_tokens_tensor = np.stack(text_token_embeds)
            assert all_tokens_tensor.shape[0] == len(token_list)

            bert_embeds_for_text_tokens_dict[instance_id] = all_tokens_tensor
        # endfor
        return bert_embeds_for_text_tokens_dict

    def get_glove_embed(self, examples, args):
        glove_embeds_for_text_tokens_dict = {}

        for (ex_index, example) in enumerate(tqdm(examples, desc="creating glove embed")):
            unique_id = example.instance_id
            index_to_word_dict = example.index_to_word_dict

            input_tokens = []
            for i, w in sorted(index_to_word_dict.items(), key=lambda x: x[0], reverse=False):
                assert isinstance(i, int)
                input_tokens.append(w)
            # endfor

            text_token_embeds = []
            for i, w in enumerate(input_tokens):
                w = w.lower()
                text_token_embeds.append(
                    self.glove_embed_handler.get_word_vector_and_masked_wordvector(w, args.input_size))
            # endfor

            all_tokens_tensor = np.stack(text_token_embeds)
            assert all_tokens_tensor.shape[0] == len(input_tokens)

            glove_embeds_for_text_tokens_dict[unique_id] = all_tokens_tensor
        # endfor

        print("Total OOV: {}".format(len(self.glove_embed_handler.out_of_vocabulary_vector_dict)))
        with open(os.path.join(args.output_dir, "OOV.txt"), mode="a") as fout:
            for k, v in self.glove_embed_handler.out_of_vocabulary_vector_dict.items():
                fout.write(k + '\n')
            # endfor

        return glove_embeds_for_text_tokens_dict

    def convert_examples_to_features(self, examples, mode):
        assert mode in ["train", "valid", "test"]
        label_list = self.get_labels()
        label_to_i_dict = {label: i for i, label in enumerate(label_list)}

        if mode == "train":
            with open(os.path.join(self.args.output_dir, "label_to_i_map.json"), mode="w") as fout:
                json.dump(label_to_i_dict, fout)
            # endwith
        # endif

        assert self.args.feature_type in {"glove", "bert"}

        unique_id_word_embed_matrix_dict = None
        if self.args.feature_type == "glove":
            unique_id_word_embed_matrix_dict = self.get_glove_embed(examples, self.args)
        # endif
        if self.args.feature_type == "bert":
            unique_id_word_embed_matrix_dict = self.get_bert_embed(examples, self.args)

        assert unique_id_word_embed_matrix_dict is not None

        feature_list = []
        for (ex_index, example) in enumerate(tqdm(examples, desc=f"creating {mode} instance feature")):
            instance_id = example.instance_id
            label_id = label_to_i_dict[example.label]

            feature = GAT_Factual_Reasoning_Feature(instance_id=instance_id,
                                                    target_verb_index=example.target_verb_index,
                                                    index_to_word_dict=example.index_to_word_dict,
                                                    index_to_lemma_dict=example.index_to_lemma_dict,
                                                    index_to_pos_dict=example.index_to_pos_dict,
                                                    word_embed_matrix=unique_id_word_embed_matrix_dict[instance_id],
                                                    directed_graph_edges=example.directed_graph_edges,
                                                    undirected_graph_edges=example.undirected_graph_edges,
                                                    directed_dep_relation_info=example.directed_dep_relation_info,
                                                    label_id=label_id)

            feature_list.append(feature)
        # endfor
        return feature_list

    def get_hypernym_glove_embed(self, synset_name_str):
        synset_name_str = synset_name_str.split(".")[0]
        synset_name_str = synset_name_str.replace("-", " ")
        synset_name_str = synset_name_str.replace("_", " ")
        synset_name_str = synset_name_str.strip()

        synset_name_token_list = synset_name_str.split()

        hypernym_token_embed_list = []
        for w in synset_name_token_list:
            w_embed = self.glove_embed_handler.get_word_vector_and_masked_wordvector(w, self.args.input_size)
            hypernym_token_embed_list.append(w_embed)
        # endfor

        hypernym_embed_avg = np.array(hypernym_token_embed_list).mean(0)

        return hypernym_embed_avg

    def get_glove_embed_with_hypernym(self, examples, args):
        glove_embeds_for_text_tokens_dict = {}

        for (ex_index, example) in enumerate(tqdm(examples, desc="creating glove embed with hypernym")):
            unique_id = example.instance_id
            index_to_word_dict = example.index_to_word_dict

            input_tokens = []
            for i, w in sorted(index_to_word_dict.items(), key=lambda x: x[0], reverse=False):
                assert isinstance(i, int)
                input_tokens.append(w)
            # endfor

            text_token_embeds = []
            for i, w in enumerate(input_tokens):
                w = w.lower()
                text_token_embeds.append(
                    self.glove_embed_handler.get_word_vector_and_masked_wordvector(w, args.input_size))
            # endfor

            for hyper_index_list, hypernym_synset_name_list in example.index_to_hypernym_list:
                hypernym_synset_name_set = set(hypernym_synset_name_list) - self.omit_general_hypernym_set

                if len(hypernym_synset_name_set) > 0:
                    # get hypernym embed
                    hypernym_embed_list = []
                    for hypernym_synset_item in hypernym_synset_name_set:
                        hypernym_item_embed = self.get_hypernym_glove_embed(hypernym_synset_item)
                        hypernym_embed_list.append(hypernym_item_embed)
                    # endfor
                    hypernym_embed_avg = np.array(hypernym_embed_list).mean(0)

                    for i in hyper_index_list:
                        text_token_embeds[i] = text_token_embeds[i] + hypernym_embed_avg
                    # endfor
                # endif
            # endfor

            all_tokens_tensor = np.stack(text_token_embeds)
            assert all_tokens_tensor.shape[0] == len(input_tokens)

            glove_embeds_for_text_tokens_dict[unique_id] = all_tokens_tensor
        # endfor

        print("Total OOV: {}".format(len(self.glove_embed_handler.out_of_vocabulary_vector_dict)))
        with open(os.path.join(args.output_dir, "OOV.txt"), mode="a") as fout:
            for k, v in self.glove_embed_handler.out_of_vocabulary_vector_dict.items():
                fout.write(k + '\n')
            # endfor

        return glove_embeds_for_text_tokens_dict

    def get_hypernym_contextual_bert_embed(self, token_list, index_list, synset_name_str):
        synset_name_str = synset_name_str.split(".")[0]
        synset_name_str = synset_name_str.replace("-", " ")
        synset_name_str = synset_name_str.replace("_", " ")
        synset_name_str = synset_name_str.strip()
        synset_name_len = len(synset_name_str.split())

        target_index = index_list[0]

        new_token_list = []
        for idx, token in enumerate(token_list):
            if idx == index_list[0]:
                new_token_list.append(synset_name_str)
                continue
            if idx in index_list[1:]:
                continue

            new_token_list.append(token)
        # endfor

        word_pieces_list = []
        word_boundaries_list = []

        for w in new_token_list:
            word_pieces = self.bert_tokenizer.tokenize(w)
            word_boundaries_list.append([len(word_pieces_list), len(word_pieces_list) + len(word_pieces)])
            word_pieces_list += word_pieces
        # endfor
        assert len(word_boundaries_list) == len(new_token_list)

        total_input_tokens = ['[CLS]'] + word_pieces_list + ['[SEP]']
        input_ids = self.bert_tokenizer.convert_tokens_to_ids(total_input_tokens)
        segment_ids = [0] * len(total_input_tokens)

        input_ids = torch.tensor([input_ids], dtype=torch.long).to(self.args.device)
        segment_ids = torch.tensor([segment_ids], dtype=torch.long).to(self.args.device)

        with torch.no_grad():
            sequence_output, _ = self.bert_model(input_ids=input_ids,
                                                 token_type_ids=segment_ids)
        # endwith

        sequence_output = sequence_output.squeeze(dim=0)
        text_piece_embeds = sequence_output[1:-1].to('cpu').numpy()

        text_token_embeds = []
        for i, w in enumerate(word_boundaries_list):
            text_token_embeds.append(text_piece_embeds[w[0]: w[1]].mean(0))
        # endfor

        target_embed = text_token_embeds[target_index]
        return target_embed

    def get_bert_embed_with_hypernym(self, examples, args):
        if self.bert_model is None and self.bert_tokenizer is None:
            from transformers import BertModel, BertTokenizer
            print("loading bert model: {} to create feature for sent tokens".format(
                args.pretrained_transformer_model_name_or_path))
            self.bert_model = BertModel.from_pretrained(args.pretrained_transformer_model_name_or_path)
            self.bert_tokenizer = BertTokenizer.from_pretrained(args.pretrained_transformer_model_name_or_path)
            self.bert_model.eval()
            self.bert_model.to(args.device)
            print(f"{args.pretrained_transformer_model_name_or_path} is successfully loaded.")

        bert_embeds_for_text_tokens_dict = {}
        for (ex_index, example) in enumerate(
                tqdm(examples,
                     desc=f"create bert embedding with hypernym from model - {args.pretrained_transformer_model_name_or_path}")):
            instance_id = example.instance_id

            index_to_word_dict = example.index_to_word_dict
            token_list = [word for index, word in index_to_word_dict.items()]

            word_pieces_list = []
            word_boundaries_list = []

            for w in token_list:
                word_pieces = self.bert_tokenizer.tokenize(w)
                word_boundaries_list.append([len(word_pieces_list), len(word_pieces_list) + len(
                    word_pieces)])
                word_pieces_list += word_pieces
            # endfor
            assert len(word_boundaries_list) == len(token_list)

            total_input_tokens = ['[CLS]'] + word_pieces_list + ['[SEP]']
            input_ids = self.bert_tokenizer.convert_tokens_to_ids(total_input_tokens)
            segment_ids = [0] * len(total_input_tokens)

            input_ids = torch.tensor([input_ids], dtype=torch.long).to(args.device)
            segment_ids = torch.tensor([segment_ids], dtype=torch.long).to(args.device)

            with torch.no_grad():
                sequence_output, _ = self.bert_model(input_ids=input_ids,
                                                     token_type_ids=segment_ids)
            # endwith

            sequence_output = sequence_output.squeeze(dim=0)
            text_piece_embeds = sequence_output[1:-1].to('cpu').numpy()  # [CLS] and [SEP] should be excluded

            text_token_embeds = []
            for i, w in enumerate(word_boundaries_list):
                text_token_embeds.append(
                    text_piece_embeds[word_boundaries_list[i][0]: word_boundaries_list[i][1]].mean(0))
            # endfor

            for hyper_index_list, hypernym_synset_name_list in example.index_to_hypernym_list:
                hypernym_synset_name_set = set(hypernym_synset_name_list) - self.omit_general_hypernym_set

                if len(hypernym_synset_name_set) > 0:
                    hypernym_embed_list = []
                    for hypernym_synset_item in hypernym_synset_name_set:
                        hypernym_item_embed = self.get_hypernym_contextual_bert_embed(token_list,
                                                                                      hyper_index_list,
                                                                                      hypernym_synset_item)
                        hypernym_embed_list.append(hypernym_item_embed)
                    # endfor

                    hypernym_embed_avg = np.array(hypernym_embed_list).mean(0)

                    for i in hyper_index_list:
                        text_token_embeds[i] = text_token_embeds[i] + hypernym_embed_avg
                    # endfor
                # endif
            # endfor

            all_tokens_tensor = np.stack(text_token_embeds)
            assert all_tokens_tensor.shape[0] == len(token_list)

            bert_embeds_for_text_tokens_dict[instance_id] = all_tokens_tensor
        # endfor
        return bert_embeds_for_text_tokens_dict

    def convert_examples_to_features_with_hypernym(self, examples, mode):
        assert mode in ["train", "valid", "test"]
        label_list = self.get_labels()
        label_to_i_dict = {label: i for i, label in enumerate(label_list)}

        if mode == "train":
            with open(os.path.join(self.args.output_dir, "label_to_i_map.json"), mode="w") as fout:
                json.dump(label_to_i_dict, fout)
            # endwith
        # endif

        assert self.args.feature_type in {"glove", "bert"}

        unique_id_word_embed_matrix_dict = None
        if self.args.feature_type == "glove":
            unique_id_word_embed_matrix_dict = self.get_glove_embed_with_hypernym(examples, self.args)
        # endif
        if self.args.feature_type == "bert":
            unique_id_word_embed_matrix_dict = self.get_bert_embed_with_hypernym(examples, self.args)

        assert unique_id_word_embed_matrix_dict is not None

        feature_list = []
        for (ex_index, example) in enumerate(tqdm(examples, desc=f"creating {mode} instance feature")):
            instance_id = example.instance_id
            label_id = label_to_i_dict[example.label]

            feature = GAT_Factual_Reasoning_Feature(instance_id=instance_id,
                                                    target_verb_index=example.target_verb_index,
                                                    index_to_word_dict=example.index_to_word_dict,
                                                    index_to_lemma_dict=example.index_to_lemma_dict,
                                                    index_to_pos_dict=example.index_to_pos_dict,
                                                    word_embed_matrix=unique_id_word_embed_matrix_dict[instance_id],
                                                    directed_graph_edges=example.directed_graph_edges,
                                                    undirected_graph_edges=example.undirected_graph_edges,
                                                    directed_dep_relation_info=example.directed_dep_relation_info,
                                                    label_id=label_id)

            feature_list.append(feature)
        # endfor
        return feature_list

    def sampling_negative_entity_in_wordnet_with_the_same_length(self, synset_str, entity_index_list):
        """
        might change the sampling strategy later
        """
        if synset_str in self.sampling_knowledge_dict:
            sample_list = self.sampling_knowledge_dict[synset_str]

            self.sampling_freq_counter[synset_str] += 1
        else:
            path = os.path.join(self.args.wordnet_synset_similarity_dict_folder, f"{synset_str}_sim.txt")
            assert os.path.exists(path)
            sample_list = self._load_synset_from_rank_list(path)

            self.sampling_knowledge_dict[synset_str] = sample_list
            self.sampling_freq_counter[synset_str] += 1
        # endif
        assert sample_list is not None

        length = len(sample_list)
        sample_list_top_k = sample_list[:int(length * 0.1)]
        sample_list_top_k_set = set(sample_list_top_k)

        original_entity_size = len(entity_index_list)
        original_list = self.wordnet_word_size_dict[original_entity_size]
        candidate_list = list(set(original_list) - sample_list_top_k_set)

        tmp_length = len(candidate_list)
        random_index = random.randint(0, tmp_length - 1)
        sampled_synset_str = candidate_list[random_index]
        return sampled_synset_str

    def _load_synset_from_rank_list(self, synset_rank_file_path):
        tmp_list = []
        with open(synset_rank_file_path, "r") as fin:
            index = -1
            for line in fin:
                index += 1
                if index == 0:
                    continue
                # endif
                line = line.strip()
                if len(line) == 0:
                    continue
                # endif
                parts = line.split()
                tmp_list.append(parts[0])
            # endfor
        # endwith
        return tmp_list

    def clean_up_sampling_knowledge_dict(self):
        print(f"sampling_knowledge_dict size: {len(self.sampling_knowledge_dict)}")
        cache_num = 100
        top_k_key = []
        if len(self.sampling_freq_counter) > cache_num:
            for k, v in sorted(self.sampling_freq_counter.items(), key=lambda x: x[1], reverse=True):
                top_k_key.append(k)
            # endfor

            top_k_key = top_k_key[:cache_num]
            delete_key = set(self.sampling_knowledge_dict.keys()) - set(top_k_key)

            print(f"delete key: {delete_key} ...")
            for k in delete_key:
                del self.sampling_knowledge_dict[k]
            # endfor
            print("done")
            print(f"sampling_knowledge_dict size: {len(self.sampling_knowledge_dict)}")
        # endif

    def replace_original_entity_with_new_of_same_word_length(self, example, new_word_list, entity_index_list,
                                                            negative_entity_synset_str):
        new_example = copy.deepcopy(example)

        new_index_to_word_dict = copy.deepcopy(example.index_to_word_dict)
        new_index_to_lemma_dict = copy.deepcopy(example.index_to_lemma_dict)
        for i, index in enumerate(entity_index_list):
            new_index_to_word_dict[index] = new_word_list[i]
            new_index_to_lemma_dict[index] = new_word_list[i]
        # endfor

        new_sent_token_list = [v for k, v in new_index_to_word_dict.items()]
        new_example.index_to_word_dict = new_index_to_word_dict
        new_example.index_to_lemma_dict = new_index_to_lemma_dict
        new_example.sent_text = " ".join(new_sent_token_list)
        new_example.instance_id = f"neg_{example.instance_id}"
        new_example.label = "novel"

        if example.entity_wordnet_info_list is not None:
            new_entity_wordnet_info_list = []
            for tmp_index_list, tmp_wordnet_info in example.entity_wordnet_info_list:
                if tmp_index_list == entity_index_list:
                    new_entity_wordnet_info_list.append([tmp_index_list, negative_entity_synset_str])
                else:
                    new_entity_wordnet_info_list.append([tmp_index_list, tmp_wordnet_info])
                # endif
            # endfor
            new_example.entity_wordnet_info_list = new_entity_wordnet_info_list
        # endif

        if example.index_to_hypernym_list is not None:
            new_index_to_hypernym_list = []
            for tmp_index_list, hypernym_info in example.index_to_hypernym_list:
                if tmp_index_list == entity_index_list:
                    new_hypernym_list = self.get_hypernym_list(negative_entity_synset_str)
                    new_index_to_hypernym_list.append([tmp_index_list, new_hypernym_list])
                else:
                    new_index_to_hypernym_list.append([tmp_index_list, hypernym_info])
                # endif
            # endfor
            new_example.index_to_hypernym_list = new_index_to_hypernym_list
        # endif

        return new_example

    def dynamically_get_positive_and_negative_batch_without_online_parsing(self, args, examples):
        print("------------------- dynamically_get_positive_and_negative_batch_without_online_parsing ----------------")
        print(
            f">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> sampling knowledge dict size: {len(self.sampling_knowledge_dict)} <<<<<<<<<<<<<<<<<<<<<<<<<<<<")
        self.clean_up_sampling_knowledge_dict()

        if self.args.feature_type == "bert":
            if self.bert_model is None and self.bert_tokenizer is None:
                from transformers import BertModel, BertTokenizer
                print("loading bert model: {} to create feature for sent tokens".format(
                    args.pretrained_transformer_model_name_or_path))
                self.bert_model = BertModel.from_pretrained(args.pretrained_transformer_model_name_or_path)
                self.bert_tokenizer = BertTokenizer.from_pretrained(args.pretrained_transformer_model_name_or_path)
                self.bert_model.eval()
                self.bert_model.to(args.device)
                print(f"{args.pretrained_transformer_model_name_or_path} is successfully loaded.")
            # endif

        negative_example_list = []
        for exp in examples:

            entity_wordnet_info_list = exp.entity_wordnet_info_list

            for entity_info in entity_wordnet_info_list:
                entity_index_list, entity_synset_str = entity_info
                if len(entity_synset_str) == 0:
                    continue

                negative_entity_synset_str = self.sampling_negative_entity_in_wordnet_with_the_same_length(
                    entity_synset_str,
                    entity_index_list)

                parts = negative_entity_synset_str.split(".")
                negative_str = parts[0]
                negative_str = negative_str.replace("_", " ")
                negative_str = negative_str.replace("-", " ")
                negative_words = negative_str.split()

                assert len(negative_words) == len(entity_index_list)

                negative_example = self.replace_original_entity_with_new_of_same_word_length(exp,
                                                                                            negative_words,
                                                                                            entity_index_list,
                                                                                            negative_entity_synset_str)
                negative_example_list.append(negative_example)
            # endfor
        # endfor

        random.shuffle(negative_example_list)

        batch_size = min(len(examples), len(negative_example_list))
        positive_example_list = examples[:batch_size]
        negative_example_list = negative_example_list[:batch_size]

        negative_feature = None
        positive_feature = None
        if args.additional_feature == "nothing":
            negative_feature = self.convert_examples_to_features(negative_example_list, "train")
            positive_feature = self.convert_examples_to_features(positive_example_list, "train")

        if args.additional_feature == "hypernym_embed":
            negative_feature = self.convert_examples_to_features_with_hypernym(negative_example_list, "train")
            positive_feature = self.convert_examples_to_features_with_hypernym(positive_example_list, "train")

        assert negative_feature is not None
        assert positive_feature is not None

        total_feature_list = []
        total_feature_list.extend(positive_feature)
        total_feature_list.extend(negative_feature)

        total_model_input = self.get_model_input(total_feature_list)
        return total_model_input

    def get_model_input(self, feature_list):
        graph_edge_list = []

        word_embed_matrix_list = []
        input_token_size_list = []
        target_mask_list = []
        label_id_list = []
        for (i, feature) in enumerate(feature_list):
            target_verb_index = int(feature.target_verb_index)

            length = len(feature.index_to_word_dict)

            word_embed_matrix = feature.word_embed_matrix
            assert word_embed_matrix.shape[0] == length

            undirected_graph_edges = feature.undirected_graph_edges
            label_id = feature.label_id

            target_mask = [False] * length
            target_mask[target_verb_index] = True

            word_embed_matrix_list.append(word_embed_matrix)
            graph_edge_list.append(np.array(undirected_graph_edges) + sum(input_token_size_list))

            target_mask_list.append([False] * sum(input_token_size_list) + target_mask)
            input_token_size_list.append(length)
            label_id_list.append(label_id)
        # endfor

        word_embed_matrix_list = np.concatenate(word_embed_matrix_list, 0)
        graph_edge_list = np.concatenate(graph_edge_list, 0)

        assert sum(input_token_size_list) == len(target_mask_list[-1])
        assert len(input_token_size_list) == len(feature_list)
        total_tokens_in_batch = sum(input_token_size_list)

        new_target_mask_list = []
        for target_mask_item in target_mask_list:
            padding_mask = [False] * (total_tokens_in_batch - len(target_mask_item))
            new_target_mask_item = target_mask_item + padding_mask
            new_target_mask_list.append(new_target_mask_item)
        # endfor

        return (word_embed_matrix_list, graph_edge_list, new_target_mask_list, label_id_list, input_token_size_list)
