"""
Reproduct translation with Amazon and Google translation APIs.
"""
# translations APIs
from scripts.translation_APIs import aws_translate
from scripts.translation_APIs import google_translate

BATCH_SIZE = 50  # define batch to proceed translation


def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]


def batch_translate(trans_function, lines, tgt_lang, src_lang = None):
    """
    Translate a list of sentences.
    Take care of batching.
    """
    translations_dicts = []
    for chunk in list(chunks(lines, BATCH_SIZE)):
        for out_dict in trans_function(chunk, tgt_lang, src_lang):
            translations_dicts.append(out_dict)
    return translations_dicts


TRANSLATION_SERVICE = {
    "google": google_translate,
    "aws": aws_translate
    # TODO: add bing and systran
}

LANGUAGE_LIST = ['ar', 'de', 'es', 'fr', 'he', 'it', 'ru']


def translate(trans_service, src, tgt, input_file, output_file):
    """
    Translation of original data in format adapted with fast_align

    :param trans_service: MT service, string
    :param src: source language, string
    :param tgt: target language, string
    :param input_file: path to input_file, string
    :param output_file: path to output_file , string
    """
    trans_function = TRANSLATION_SERVICE[trans_service]
    lines = [line.strip() for line in open(input_file, encoding="utf8")]
    out_dicts = batch_translate(trans_function, lines, tgt, src)
    with open(output_file, "w", encoding="utf8") as file:
        for out_dict in out_dicts:
            # this type of output "{} ||| {}" is necessary to proceed the alignment
            file.write("{} ||| {}\n".format(out_dict["input"],
                                            out_dict["translatedText"]))


if __name__ == '__main__':
    print('Executing translation ...')
    for trans_service in ["aws"]:
        for target in LANGUAGE_LIST:
            translate(trans_service,
                      'en',  # language source is english
                      target,
                      '../data/en.txt',
                      '../translations/{}/en-{}.txt'.format(trans_service, target))
    print('Translations completed')

