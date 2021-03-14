"""
Main file : Gender Bias Evaluation
"""
import pandas as pd
import os

import pickle

from evaluation import evaluate_bias

from load_alignments import LANGUAGE_PREDICTOR, get_translated_professions, align_bitext_to_ds


os.chdir('scripts/')


def listify_file(ds=None, bi=None, align=None):
	"""
	Transform path into list
	"""
	if ds:
		ds = [line.strip().split("\t") for line in open(ds, encoding="utf8")]
	if bi:
		bi = [line.strip().split(" ||| ") for line in open(bi, encoding="utf8")]
	if align:
		align = [line for line in open(align)]
	return ds, bi, align


def procede_evaluation(ds, bi, align, lang):
	gender_predictor = LANGUAGE_PREDICTOR[lang]()
	bitext = align_bitext_to_ds(bi, ds)
	translated_profs, tgt_inds = get_translated_professions(align, ds, bitext)
	assert (len(translated_profs) == len(tgt_inds))
	target_sentences = [tgt_sent for (ind, (src_sent, tgt_sent)) in bitext]
	gender_predictions = [gender_predictor.get_gender(prof, translated_sent, entity_index, ds_entry)
						  for prof, translated_sent, entity_index, ds_entry
						  in zip(translated_profs,
								 target_sentences,
								 map(lambda ls:min(ls, default=-1), tgt_inds),
								 ds)]
	d = evaluate_bias(ds, gender_predictions)
	return d


MT_LIST = ['bing', 'google', 'systran', 'aws']  # no ukranian translation with amazon
LANGUAGE_LIST_ukless = ['fr', 'es', 'it', 'ar', 'he', 'ru']
LANGUAGE_LIST_all = ['fr', 'es', 'de', 'it', 'ar', 'he', 'ru', 'uk']

with open('../translations/_aligns/pro-anti_indexes.pkl', 'rb') as file:
	pro_ind, anti_ind = pickle.load(file)


def main_function(tab, ds_fn):
	"""
	Compute all results
	"""
	ds, _, _ = listify_file(ds=ds_fn)
	for MT in MT_LIST:
		if MT == 'aws':
			LANGUAGE_LIST = LANGUAGE_LIST_ukless
		else:
			LANGUAGE_LIST = LANGUAGE_LIST_all
		df = pd.DataFrame(columns=['acc', 'f1_female', 'f1_male'])
		for lang in LANGUAGE_LIST:
			bi_fn = '../translations/{}/en-{}.txt'.format(MT, lang)
			al_fn = '../translations/_aligns/{}/forwarden-{}.align'.format(MT, lang)
			_, bi, align = listify_file(bi=bi_fn, align=al_fn)
			if tab == 'pro':
				ds_ = [ds[i] for i in pro_ind]
				bi_ = [bi[i] for i in pro_ind]
				align_ = [align[i] for i in pro_ind]
			if tab == 'anti':
				ds_ = [ds[i] for i in anti_ind]
				bi_ = [bi[i] for i in anti_ind]
				align_ = [align[i] for i in anti_ind]
			if tab == 'all':
				ds_ = ds.copy()
				bi_ = bi.copy()
				align_ = align.copy()

			d = procede_evaluation(ds_, bi_, align_, lang)
			print('Gender bias valuation with MT {} in {}'.format(MT, lang), d)
			temp_d = {k: v for k, v in d.items() if k in ['acc', 'f1_female', 'f1_male']}
			temp = pd.DataFrame(temp_d, index=[lang])
			df = pd.concat([df, temp])
		df.to_csv('../results/eval_{}_{}.csv'.format(tab, MT))


def compute_final_dataframe():
	for MT in MT_LIST:
		df = pd.DataFrame(columns=['acc', 'DeltaG', 'DeltaS'])
		suball = pd.read_csv("../results/eval_all_{}.csv".format(MT), index_col=0)
		subpro = pd.read_csv("../results/eval_pro_{}.csv".format(MT), index_col=0)
		subanti = pd.read_csv("../results/eval_anti_{}.csv".format(MT), index_col=0)
		df['acc'] = suball['acc']
		df['DeltaG'] = abs(suball['f1_male'] - suball['f1_female'])
		df['DeltaS'] = abs(subpro[['f1_male', 'f1_female']].max(axis=1) - subanti[['f1_male', 'f1_female']].max(axis=1))
		df.to_csv('../results/final_{}.csv'.format(MT))


if __name__ == "__main__":
	ds_fn = '../data/en.txt'
	
	print('generate results for all dataset')
	main_function('all', ds_fn)
	print('generate results for anti-stereotypical dataset')
	main_function('anti', ds_fn)
	print('generate results for pro-stereotypical dataset')
	main_function('pro', ds_fn)
	print("Generate final dataframe...")
	compute_final_dataframe()
	print('Done !')







