import os
import logging
import time
from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec
import numpy as np
import wget


class GloveEmbed(object):
    def __init__(self, glove_data_folder, download_type=None):
        self.glove_data_folder = glove_data_folder
        if not os.path.exists(self.glove_data_folder):
            os.makedirs(self.glove_data_folder)
        # endif
        self._create_file_path_dict()
        self.gensim_format_glove_embed_file = None

        if not self._check_exists(download_type):
            self._download(download_type)
        # endfor

        self.out_of_vocabulary_vector_dict = {}

    def _create_file_path_dict(self):
        path_dict = {"glove_6B_50d": os.path.join(self.glove_data_folder, "glove.6B.50d.txt"),
                     "glove_6B_100d": os.path.join(self.glove_data_folder, "glove.6B.100d.txt"),
                     "glove_6B_200d": os.path.join(self.glove_data_folder, "glove.6B.200d.txt"),
                     "glove_6B_300d": os.path.join(self.glove_data_folder, "glove.6B.300d.txt"),
                     "glove_42B": os.path.join(self.glove_data_folder, "glove.42B.300d.txt"),
                     "glove_840B": os.path.join(self.glove_data_folder, "glove.840B.300d.txt")}
        self.path_dict = path_dict
        return path_dict

    def _check_exists(self, download_type):
        embed_path = self.path_dict[download_type]
        if os.path.exists(embed_path):
            return True
        else:
            return False
        # endif

    def _download(self, download_type=None):
        links_dict = {"glove.6B.zip": "http://nlp.stanford.edu/data/glove.6B.zip",
                      "glove.42B.300d.zip": "http://nlp.stanford.edu/data/glove.42B.300d.zip",
                      "glove.840B.300d.zip": "http://nlp.stanford.edu/data/glove.840B.300d.zip"}

        if download_type is None:
            for file_name, url in links_dict.items():
                print("Downloading {} ...".format(file_name))
                downloaded_file_path = os.path.join(self.glove_data_folder, file_name)
                if not os.path.exists(downloaded_file_path):
                    wget.download(url, downloaded_file_path)
                    print("downloaded")

                    cwd = os.getcwd()
                    os.chdir(self.glove_data_folder)
                    print("unzip {}".format(file_name))
                    os.system("unzip {}".format(file_name))
                    print("unzip done")
                    os.chdir(cwd)
            # endfor
        else:
            option = None
            if "840B" in download_type:
                option = "glove.840B.300d.zip"
            elif "42B" in download_type:
                option = "glove.42B.300d.zip"
            elif "6B" in download_type:
                option = "glove.6B.zip"
            else:
                print("glove embedding option is wrong: {}".format(download_type))
                exit(-1)
            # endif

            print("Downloading {} ...".format(option))
            downloaded_file_path = os.path.join(self.glove_data_folder, option)
            if not os.path.exists(downloaded_file_path):
                wget.download(links_dict[option], downloaded_file_path)
                print("downloaded")

                cwd = os.getcwd()
                os.chdir(self.glove_data_folder)
                print("unzip {}".format(option))
                os.system("unzip {}".format(option))
                print("unzip done")
                os.chdir(cwd)
            # endif

    def _convert_original_glove_embed_to_gensim_format(self, glove_original_file_path, gensim_format_glove_embed_file):

        if os.path.exists(gensim_format_glove_embed_file):
            print("{} is already exists.".format(gensim_format_glove_embed_file))
            return
        else:
            print("Converting {} into gensim format as {}".format(glove_original_file_path,
                                                                  gensim_format_glove_embed_file))

            glove2word2vec(glove_original_file_path, gensim_format_glove_embed_file)
            logging.info("converted")
        # endif

    def load_model(self, glove_embed_name):
        assert glove_embed_name in self.path_dict
        glove_file_path = self.path_dict[glove_embed_name]
        gensim_format_glove_embed_file = glove_file_path.replace(".txt", "_gensim.txt")
        if not os.path.exists(gensim_format_glove_embed_file):
            self._convert_original_glove_embed_to_gensim_format(glove_file_path, gensim_format_glove_embed_file)
        # endif
        logging.info("load model ...")
        begin = time.time()
        model = KeyedVectors.load_word2vec_format(gensim_format_glove_embed_file, unicode_errors='strict')
        time_length = time.time() - begin
        logging.info(f"model takes {time_length * 1.0 / 60} mins to get loaded.")
        self.model = model
        return model

    def get_word_vector(self, word, word_embed_size):
        word_vectors = self.model.wv
        if word not in word_vectors:
            print("OOV word : {}".format(word))
            if word in self.out_of_vocabulary_vector_dict:
                return self.out_of_vocabulary_vector_dict[word]
            else:
                random_embed = np.random.rand(word_embed_size)
                self.out_of_vocabulary_vector_dict[word] = random_embed
                return self.out_of_vocabulary_vector_dict[word]
            # endif
        else:
            return word_vectors[word]

    def get_word_vector_and_masked_wordvector(self, word, word_embed_size):
        word_vectors = self.model.wv
        if word not in word_vectors:
            if word in self.out_of_vocabulary_vector_dict:
                return self.out_of_vocabulary_vector_dict[word]
            else:
                if word == "[mask]":
                    embed = np.zeros(word_embed_size)
                    self.out_of_vocabulary_vector_dict[word] = embed
                    return self.out_of_vocabulary_vector_dict[word]
                else:
                    random_embed = np.random.rand(word_embed_size)
                    self.out_of_vocabulary_vector_dict[word] = random_embed
                    return self.out_of_vocabulary_vector_dict[word]
            # endif
        else:
            return word_vectors[word]
