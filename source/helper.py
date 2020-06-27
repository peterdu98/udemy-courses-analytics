''' This file is to make helper functions for the entire project.
	
	Functionalities
	---------------
	Data exploration - Tag: `check`, `describe`, `get`
	Feature engineering - Tag: `convert`, `merge`, `make`, `preprocess`

'''
import pandas as pd
import numpy as np
import copy
import nltk
import string

######################
###### CONSTANT ######
######################
STOPWORDS = set(nltk.corpus.stopwords.words('english'))
PUNCT = string.punctuation

##############################
###### DATA EXPLORATION ######
##############################
def check_nan(df):
	return df.isna().sum(axis=0)

def check_text_length(df, column, length, is_index=False):
	# Getting text that is less than the required length of words
	tokenize_text = df[column].apply(lambda text: nltk.word_tokenize(text))
	tokens_length = tokenize_text.apply(lambda tokens: len(tokens) <= length)
	ind = np.nonzero(tokens_length.values)[0]

	if is_index:
		return ind

	return df.iloc[ind]

def describe_freq(df, column, times):
	# Initialising the first counting
	res = df[column].value_counts()
	
	# Next counting operations
	for i in range(1, times):
		res = res.value_counts()
	return res

def describe_crosstab(df, x_col, y_col):
	# Copying an object
	temp = df.copy()

	# Converting features into the binary form
	temp[x_col] = temp[x_col] > temp[x_col].mean()
	temp[y_col] = temp[y_col] > temp[y_col].mean()

	# Applying crosstab into the converted feature
	res = pd.crosstab(temp[x_col], temp[y_col])
	res.columns = ['low_{}'.format(y_col), 'high_{}'.format(y_col)]
	res.index = ['low_{}'.format(x_col), 'high_{}'.format(x_col)]

	return res

def get_rows_by_mean(df, column, option=None):
	# Copying an object
	res = df.copy()

	# Filtering the feature
	if option == "more":
		res = res[res[column] > np.mean(res[column])]
	elif option == "less":
		res = res[res[column] < np.mean(res[column])]

	return res

def get_top_topics(df, columns, phrase_len, top=None, by="freq"):
	# Initialisation
	res = dict()

	# Generate staistical information
	if by == "freq":
		# Getting the frequency for each topic
		for topics in df[columns].values:
			topics = topics[0].split(", ")
			for topic in topics:
				tokens = topic.split()
				if len(tokens) == phrase_len:
					if topic not in res:
						res[topic] = 1
					else:
						res[topic] += 1
	elif by in ["sub", "reviews"]:
		# Getting number of subscribers/reviews for each topic
		for num, topics in df[columns].values:
			topics = topics.split(", ")
			for topic in topics:
				tokens = topic.split()
				if len(tokens) == phrase_len:
					if topic not in res:
						res[topic] = num
					else:
						res[topic] += num

	# Sorting
	res = sorted(res.items(), key=lambda x: x[1], reverse=True)

	if top is not None:
		return res[:top:]
	return res


#################################
###### FEATURE ENGINEERING ######
#################################
def convert_bool_to_int(row):
	''' This is an inline function that is used to convert each row in the is_paid feature into the binary integer value. '''
	if row == 'true':
		return 1
	elif row == 'false':
		return 0
	
	return np.nan

def convert_url_to_string(df, column):
	res = df.copy()
	urls_hyphen = res[column].apply(lambda url: url.split('/')[-2])
	urls_string = urls_hyphen.apply(lambda url: " ".join(url.split("-")))
	res[column] = urls_string

	return res

def convert_free_price(df, column):
	# Copying an object
	res = df.copy()

	# Converting the price feature into a numeric feature
	res[column] = res[column].apply(lambda price: price if price != 'Free' else '0')
	res[column] = res[column].astype(np.int32)

	return res

def convert_duration(row):
	''' This is an inline function that is used to convert each row in the duration feature  into the floating value. '''

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
	# Copying an object
	res = df.copy()

	# Getting year from the timestamp feature
	res[column] = res[column].apply(lambda time: time.split("-")[0])
	res['published_year'] = res[column].astype(np.int16)
	res.drop(column, axis=1, inplace=True)

	return res

def make_left_to_right(df, left_col, right_col, ind):
	# Copying an object
	res = df.copy()

	# Assignment operation
	res.loc[ind, left_col] = res.loc[ind, right_col]
	
	return res

def merge_duplicate_row(df, id_column):
	# Initliasation
	unique_id = set()
	unique_index = []
	course_ids = df[id_column].values

	# Getting unique course ids and indexes
	for ind, course_id in enumerate(course_ids):
		if course_id not in unique_id:
			unique_id.add(course_id)
			unique_index.append(ind)

	return df.iloc[unique_index]

def preprocess_text(df, column, new_col):
	# Copying an object
	res = df.copy()

	# Initialisation
	ignored_words = set(["learn", "learned", "learning", "build", "building", "scratch", "create", "how", "using", "maximize", "course", "beginner", "beginners", "without", "easy", "pro", "hours", "minutes", "play", "hour", "levels", "level", "step", "basics", "complete", "weird", "parts", "introduction", "brief", "week"])
	ignored_words = ignored_words.union(STOPWORDS)

	# Lowercase the feature
	res[new_col] = res[column].apply(lambda text: text.lower())

	# Remove punctuation
	res[new_col] = res[new_col].apply(lambda text: text.translate(text.maketrans('', '', PUNCT)))

	# Remove ignored words
	res[new_col] = res[new_col].apply(lambda text: text.split()) \
								.apply(lambda words: " ".join([word for word in words if word not in ignored_words]))

	# Drop empty string
	res[new_col] = res[new_col].replace('', np.nan)
	res.dropna(subset=[new_col], inplace=True)

	# Parse noun phrases for the text
	res[new_col] = res[new_col].apply(lambda text: _parse_text(text))

	return res

#############################
###### UTILITY METHODS ######
#############################

def _parse_text(text):
	# Initialisation
	res = []
	grammar = '''
		NP: {(<JJ>|<NN>*)<NN>}
	'''

	# Assign a tag for each text
	temp = nltk.pos_tag(text.split())

	# Normalize tags ("NN", "NN-PL", "NNS" -> "NN")
	sent = []
	for word, tag in temp:
		if tag == 'NP-TL' or tag == 'NP':
			sent.append(word, 'NNP')
			continue
		if tag.endswith('-TL'):
			sent.append((word, tag[:-3]))
			continue
		if tag.endswith('S'):
			sent.append((word, tag[:-1]))
			continue
		sent.append((word, tag))

	# Build a parser
	cp = nltk.RegexpParser(grammar)
	tree = cp.parse(sent)

	for subtree in tree.subtrees():
		if subtree.label() == "NP":
			res.append(" ".join(set([x[0] for x in subtree])))

	return ", ".join(res)
