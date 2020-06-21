''' This file is to make helper functions for the entire project.
	
	Functionalities
	---------------
	Data exploration - Tag: `check`, `describe`
	Feature engineering - Tag: `convert`, `merge`, `make`

'''
import pandas as pd
import numpy as np
import copy
import nltk

def check_nan(df):
	return df.isna().sum(axis=0)

def check_text_length(df, column, length, is_index=False):
	tokenize_text = df[column].apply(lambda text: nltk.word_tokenize(text))
	tokens_length = tokenize_text.apply(lambda tokens: len(tokens) <= length)
	ind = np.nonzero(tokens_length.values)[0]

	if is_index:
		return ind

	return df.iloc[ind]

def convert_bool_to_int(row):
	if row == 'true':
		return 1
	elif row == 'false':
		return 0
	else:
		return np.nan

def convert_url_to_string(df, column):
	res = df.copy()
	urls_hyphen = res[column].apply(lambda url: url.split('/')[-2])
	urls_string = urls_hyphen.apply(lambda url: " ".join(url.split("-")))
	res[column] = urls_string

	return res

def convert_free_price(df, column):
	res = df.copy()
	res[column] = res[column].apply(lambda price: price if price != 'Free' else '0')
	res[column] = res[column].astype(np.int32)

	return res

def convert_duration(row):
	if row != '0':
		hour, vocab = row.split()
		if vocab == "mins":
			return float(hour)/60
		elif vocab == "hour" or "hours":
			return float(hour)
		else:
			return float(hour) * 2.5 / 60

	return np.nan

def convert_published_time(df, column):
	res = df.copy()
	res[column] = res[column].apply(lambda time: time.split("-")[0])
	res['published_year'] = res[column].astype(np.int16)
	res.drop(column, axis=1, inplace=True)

	return res

def make_left_to_right(df, left_col, right_col, ind):
	res = df.copy()
	res.loc[ind, left_col] = res.loc[ind, right_col]
	return res

def describe_freq(df, column, times):
	res = df[column].value_counts()
	for i in range(1, times):
		res = res.value_counts()
	return res

def describe_crosstab(df, x_col, y_col):
	temp = df.copy()
	temp[x_col] = temp[x_col] > temp[x_col].mean()
	temp[y_col] = temp[y_col] > temp[y_col].mean()

	res = pd.crosstab(temp[x_col], temp[y_col])
	res.columns = ['low_{}'.format(y_col), 'high_{}'.format(y_col)]
	res.index = ['low_{}'.format(x_col), 'high_{}'.format(x_col)]

	return res


def merge_duplicate_row(df, id_column):
	unique_id = set()
	unique_index = []
	course_ids = df[id_column].values

	for ind, course_id in enumerate(course_ids):
		if course_id not in unique_id:
			unique_id.add(course_id)
			unique_index.append(ind)

	return df.iloc[unique_index]