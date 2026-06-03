def extract_info(address):

    if not isinstance(address, str):
        return pd.Series({'state': None, 'zip': None})

    lines = [x.strip() for x in address.split('\n')]

    state = None
    zip_code = None

    for i, line in enumerate(lines):

        if re.fullmatch(r'[A-Z]{2}', line):
            state = line

            if i + 1 < len(lines):
                if re.fullmatch(r'\d{5}', lines[i + 1]):
                    zip_code = lines[i + 1]

            break

    return pd.Series({'state': state, 'zip': zip_code})

print(df_bbg['entity_address'].iloc[0])
