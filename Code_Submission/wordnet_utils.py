from nltk.corpus import wordnet as wn
import os


class WordNetUtils:

    @staticmethod
    def synset_list_of_entity_in_wordnet(word_list):
        synset_list = None
        boundary_index = 0
        while boundary_index < len(word_list):
            entity_str = "_".join(word_list[boundary_index:])
            synset_list = wn.synsets(entity_str)

            if len(synset_list) > 0:
                return (synset_list, boundary_index)
            else:
                boundary_index += 1

        return None

    @staticmethod
    def find_similarity_between_synsets(synset_obj_1, synset_obj_2):
        return synset_obj_1.wup_similarity(synset_obj_2)

    @staticmethod
    def find_all_synset_similarity_ranking_list_multiprocess(wordnet_synset_similarity_dict_folder,
                                                             target_synset_str_list):

        for target_synset_str in target_synset_str_list:
            synset_path = os.path.join(wordnet_synset_similarity_dict_folder, f"{target_synset_str}_sim.txt")

            target_synset_obj = wn.synset(target_synset_str)
            similarity_list = []
            all_noun_synset_list = list(wn.all_synsets('n'))
            for tmp_synset in all_noun_synset_list:
                similarity_score = WordNetUtils.find_similarity_between_synsets(tmp_synset, target_synset_obj)
                similarity_list.append([tmp_synset._name, similarity_score])
            # endfor

            sorted_similarity_list = sorted(similarity_list, key=lambda x: x[1], reverse=True)

            with open(synset_path, mode="w") as fout:
                for synset_name, score in sorted_similarity_list:
                    fout.write(f"{synset_name}\t{score}\n")
            # endfor
        # endfor

    @staticmethod
    def find_all_synset_similarity_ranking_list_single_synset(wordnet_synset_similarity_dict_folder,
                                                              target_synset_str):

        synset_path = os.path.join(wordnet_synset_similarity_dict_folder, f"{target_synset_str}_sim.txt")

        target_synset_obj = wn.synset(target_synset_str)
        similarity_list = []
        all_noun_synset_list = list(wn.all_synsets('n'))
        for tmp_synset in all_noun_synset_list:
            similarity_score = WordNetUtils.find_similarity_between_synsets(tmp_synset, target_synset_obj)
            similarity_list.append([tmp_synset._name, similarity_score])
        # endfor

        sorted_similarity_list = sorted(similarity_list, key=lambda x: x[1], reverse=True)

        with open(synset_path, mode="w") as fout:
            for synset_name, score in sorted_similarity_list:
                fout.write(f"{synset_name}\t{score}\n")
        # endfor
