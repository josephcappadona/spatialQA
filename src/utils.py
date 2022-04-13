def clean_model_name(model_name):
    return model_name.replace('/', '-')

    
def append_to_tsv(tsv_writer, gen):
    
    for (p, h), e, (r_type, fn_name) in gen:
        print(*[p, h, e, r_type, fn_name])
        tsv_writer.writerow([p, h, e, r_type, fn_name])