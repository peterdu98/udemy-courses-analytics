''' This file is to make helper functions for the entire project.
	
	Functionalities
	---------------
	Data exploration - Tag: `check`, `describe`
	Feature engineering - Tag: `convert`

'''
import pandas as pd
import numpy as np

def check_nan(df):
	return df.isna().sum(axis=0)

def check_length(df, column):
	pass

def check_url_domain(df, column):
	pass

def describe_freq(df, column):
	pass

def describe_stat(df, columns, methods):
	pass

def convert_type(df, np_type, columns):
	pass
