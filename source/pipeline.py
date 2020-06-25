''' This file is to make different pipeline for transforming features '''
import pandas as pd
import numpy as np
from . import helper

def clean_data(data):
	# Make a deep copy
	df = data.copy()

	# Merge duplicate course_id
	df = helper.merge_duplicate_row(df, 'course_id')
	df['course_id'] = df['course_id'].astype(np.int32)

	# Transform is_paid into binary integer
	df['is_paid'] = df['is_paid'].apply(lambda x: x.lower()) \
										.apply(lambda x: helper.convert_bool_to_int(x))
	df.dropna(subset=['is_paid'], inplace=True)
	df['is_paid'] = df['is_paid'].astype(np.int8)

	# Transform url into string
	df = helper.convert_url_to_string(df, 'url')

	# Make URL string to be a title for short title courses (length <= 2)
	ind = helper.check_text_length(df, 'course_title', 2, is_index=True)
	df = helper.make_left_to_right(df, 'course_title', 'url', ind)
	df.drop('url', axis=1, inplace=True)

	# Price feature
	df = helper.convert_free_price(df, "price")

	# Content duration feature
	df['content_duration'] = df['content_duration'].apply(lambda duration: helper.convert_duration(duration))
	df['content_duration'] = df['content_duration'].astype(np.float32)

	# Published time
	df = helper.convert_published_time(df, 'published_timestamp')

	return df
