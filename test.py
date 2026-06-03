import pandas as pd
import re

def extract_state_zip(df, address_column):

    def extract_info(address):

        if pd.isna(address):
            return pd.Series([None, None])

        lines = [x.strip() for x in str(address).split('\n')]

        state = None
        zip_code = None

        for line in lines:

            # State code (TX, AZ, CA, etc.)
            if re.fullmatch(r'[A-Z]{2}', line):
                state = line

            # ZIP code
            if re.fullmatch(r'\d{5}', line):
                zip_code = line

        return pd.Series([state, zip_code])

    df[['state', 'zip']] = df[address_column].apply(extract_info)

    return df

df_bbg = extract_state_zip(df_bbg, 'entity_address')
