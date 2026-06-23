val_ids = df_val.select("master_party_smun_identifier").distinct()
final_ids = df_final_selected.select("master_party_smun_identifier").distinct()

extra_ids = final_ids.subtract(val_ids)
missing_ids = val_ids.subtract(final_ids)

display(extra_ids)
display(missing_ids)


print("df_val unique:", val_ids.count())
print("df_final unique:", final_ids.count())
print("extra ids:", extra_ids.count())
print("missing ids:", missing_ids.count())

df_final_selected = df_final_selected.join(val_ids, on="master_party_smun_identifier", how="inner")
