"""
Translation APIs for google, amazon, bing and Systran
"""
import html
# from bing_translator import Bing

import boto3  # API kit for AWS
from google.cloud import translate_v2  # need install google-cloud-translate

# TODO: add systran_translation_api from https://github.com/SYSTRAN/translation-api-python-client
# import systran_translation_api
# import systran_translation_api.configuration


def aws_translate(sents, target_language, source_language):
    """
    Run AWS translate on a batch of sentences.
    """
    AWS_TRANSLATE_CLIENT = boto3.client(service_name='translate',
                                        use_ssl=True,
                                        region_name='eu-west-2')  # setting up translation API #noqa
    trans = []
    for sent in sents:
        cur_trans = {}
        cur_result = AWS_TRANSLATE_CLIENT.translate_text(Text=sent,
                                                         SourceLanguageCode=source_language,
                                                         TargetLanguageCode=target_language)
        cur_trans["translatedText"] = html.unescape(cur_result["TranslatedText"])
        cur_trans["input"] = sent
        trans.append(cur_trans)
    return trans


def google_translate(sents, target_language, source_language):
    """
    Run google translate on a batch of sentences.
    """
    GOOGLE_TRANSLATE_CLIENT = translate_v2.Client()  # initialize translator
    trans = GOOGLE_TRANSLATE_CLIENT.translate(sents,
                                              source_language=source_language,
                                              target_language=target_language)  # translate batch of sentences

    for out_dict in trans:
        out_dict["translatedText"] = html.unescape(out_dict["translatedText"])
    return trans

# ----------------------------------------------------------------------------------------------
# NOT WORKING FUNCTIONS


def bing_translate(sents, target_language, source_language):
    """
    Doesn't work yet.
    """
    # TODO: this id and secrets are available when creating an app with Microsoft Azure
    # see the following url :
    # https://stackoverflow.com/questions/41154278/client-id-and-secret-secret-for-translate-api  # noqa
    client_id = "<My-Client-Id>"  # not found
    client_secret = "<My-Client-Secret>"  # not found

    translator = Bing(client_id, client_secret) # initialize translator
    trans = []
    for sent in sents:
        out_dict = {}
        phrase_translated = translator.translate("Hello World",
                                                 to_lang=target_language,
                                                 from_lang=source_language)  # translating phrase
        out_dict["translatedText"] = phrase_translated
        out_dict["input"] = sent
        trans.append(out_dict)

    return trans


class Systran:
    """
    Systran wrapper. Doesn't work yet
    """
    def __init__(self, api_key_file):
        """
        Init systran's API
        """
        systran_translation_api.configuration.load_api_key(api_key_file)
        self.api_client = systran_translation_api.ApiClient()
        self.translation_api = systran_translation_api.TranslationApi(self.api_client)

    def translate(self, sents, tgt_lang, src_lang):
        """
        Translate a batch of sentences.
        """
        trans_result = self.translation_api.translation_text_translate_get(source = src_lang,
                                                                           target = tgt_lang,
                                                                           input = sents)

        translated_sents = [cur_output.output
                            for cur_output in trans_result.outputs]
        return translated_sents



