import unittest
import pickle

from scripts import SubForum, SubForumStats, get_all_words

'''
import json
import pandas as pd
def json_to_pandas(path_json):
    with open(path_json, 'r') as file:
        temp = json.load(file)
    return pd.DataFrame.from_dict(temp, orient='index')
questions = json_to_pandas('tests/questions.json')'''


class SubForumTest(unittest.TestCase):
    def setUp(self):
        self.object = None

    @unittest.skip
    def test_init(self):
        self.object = SubForum('./questions.json',
                               './answers.json')

    def test_discard_punctuation(self):
        self.object = SubForum('./questions.json',
                                    './answers.json')
        self.object._cleaning()
        # print(self.object.questions['body'][0])

    def test_expand_contractions(self):
        self.object = SubForum('./questions.json',
                                    './answers.json')
        self.object.expand_contractions()
        # print(self.object.questions['body'][0])

    def test_tokenize(self):
        self.object = SubForum('./questions.json',
                                    './answers.json')
        self.object.link_cleaning()
        self.object._cleaning()
        self.object.expand_contractions()
        self.object.tokenize()
        # print(self.object.questions['body'][0])

    def test_lemmatize(self):
        self.object = SubForum('./questions.json',
                                    './answers.json')
        self.object.link_cleaning()
        self.object._cleaning()
        self.object.expand_contractions()
        self.object.tokenize()
        self.object.lemmatize()
        # print(self.object.questions['body'][0])

    def test_link_cleaning(self):
        self.object = SubForum('./questions.json',
                               './answers.json')
        self.object.link_cleaning()
        print(self.object.answers['body'][0])

    def test__cleaning(self):
        self.object = SubForum('./questions.json',
                               './answers.json')
        self.object._cleaning()
        print(self.object.questions['body'][0])

    def test_pre_processing(self):
        self.object = SubForum('./questions.json',
                                    './answers.json')
        self.object.pre_processing()
        # print(self.object.questions['body'][0])
        with open('../data/data_preprocess/android_test.pkl', 'wb') as file:
            pickle.dump(self.object, file)

    def test_change_ids(self):
        self.object = SubForum('./questions.json',
                               './answers.json')
        print(self.object.answers['parentid'])
        self.object.change_ids('z')
        print(self.object.answers['parentid'])


class SubForumStatsTest(unittest.TestCase):
    def setUp(self):
        self.object = None

    def test_count_answers(self):
        self.object = SubForumStats('./questions.json',
                                    './answers.json')
        self.object._preprocessing()
        self.object.count_answers()
        # print(self.object.questions['nb_answers'])

    def test_count_words_questions(self):
        self.object = SubForumStats('./questions.json',
                                    './answers.json')
        self.object._preprocessing()
        self.object.count_words_questions()
        # print(self.object.questions['nb_words'])

    def test_count_words_threads(self):
        self.object = SubForumStats('./questions.json',
                                    './answers.json')
        self.object._preprocessing()
        self.object.count_words_threads()
        # print(self.object.questions['nb_words_threads'])

    def test_count_duplicates(self):
        self.object = SubForumStats('./questions.json',
                                    './answers.json')
        self.object._preprocessing()
        self.object.count_duplicates()
        # print(self.object.questions['nb_dups'])


class ModuleFunctionTest(unittest.TestCase):
    def test_get_all_words(self):
        self.object = SubForum('./questions.json',
                               './answers.json')
        # delete columns
        self.object.pre_processing()
        words = get_all_words(self.object)
        print(words)


if __name__ == '__main__':
    unittest.main()
