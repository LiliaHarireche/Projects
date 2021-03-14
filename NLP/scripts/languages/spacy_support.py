"""
create class SpacyPredictor for spanish, French, and Italian languages with two functions:
    _get_gender() to predict gender whithout using cache, and
     get_gender() to predict gender of an input profession
     which is retrieved according to the function: _get_gender().
"""
from collections import Counter

import spacy

from util import GENDER, SPACY_GENDER_TYPES, get_gender_from_token


class SpacyPredictor:
    """
    Class for spaCy supported languages.
    These seem to include:
    Spanish, French, and Italian.
    """
    def __init__(self, lang: str):
        """
        Init spacy for the specified language code.
        """
        assert lang in ["es", "fr", "it"]
        spacy_dict = {
            'fr': "fr_core_news_sm",
            'it': "it_core_news_sm",
            'es': "es_core_news_sm"
        }
        self.lang = lang
        self.cache = {}    # Store calculated professions genders
        self.nlp = spacy.load(spacy_dict[self.lang])

    def get_gender(self, profession: str, translated_sent=None, entity_index=None, ds_entry=None) -> GENDER:
        """
        Predict gender of an input profession.
        """
        if translated_sent == "Il venditore ha parlato con l'insegnante e l'ha invitata a cena.":
            pass
        if profession not in self.cache:
            self.cache[profession] = self._get_gender(profession) # Store calculated professions genders in cache using _get_gender() function # noqa

        return self.cache[profession]

    def _get_gender(self, profession: str) -> GENDER:
        """
        Predict gender, without using cache
        """
        if not profession.strip():
            # Empty string
            return GENDER.unknown

        toks = self.nlp(profession)
        gender_from_tok = [get_gender_from_token(token) for token in toks]
        observed_genders = [gender for gender in gender_from_tok
                            if gender is not None]
        if not observed_genders:
            # No observed gendered words - return unknown
            return 'Unk'

        if GENDER.female in observed_genders:
            return 'Fem'

        # Return the most commonly observed gender
        return Counter(observed_genders).most_common()[0][0]


if __name__ == "__main__":
    ds = ['female', 1, 'The developer argued with the designer because she did not like the design.', 'developer']
    tr_sent_fr = "Le développeur a discuté avec le concepteur parce qu'elle n'aimait pas le design."
    p = SpacyPredictor(lang="fr")
    pred_fr = p.get_gender('développeur', tr_sent_fr)
    print(pred_fr)
    print("FRENCH TEST DONE")

    ds = ['female', 1, 'The developer argued with the designer because she did not like the design.', 'developer']
    tr_sent_ru = 'Lo sviluppatore ha discusso con il designer perché non le piaceva il design.'
    p = SpacyPredictor(lang="it")
    pred_it = p.get_gender('sviluppatore', tr_sent_ru)
    print(pred_it)
    print("ITALIAN TEST DONE")

    ds = ['female', 1, 'The developer argued with the designer because she did not like the design.', 'developer']
    tr_sent_es = 'El desarrollador discutió con el diseñador porque no le gustaba el diseño.'
    p = SpacyPredictor(lang="es")
    pred_es = p.get_gender('desarrollador', tr_sent_ru, 0, ds)
    print(pred_es)
    print("SPANISH TEST DONE")
