from dataclasses import dataclass
@dataclass 
class Config: 
    batch_size =  8,
    num_epochs =  1,
    lr =  10**-4,
    seq_len =  350,
    d_model =  512,
    datasource =  'opus_books',
    lang_src =  "en",
    lang_tgt =  "it",
    model_folder =  "weights",
    model_basename =  "tmodel_",
    preload =  "latest",
    tokenizer_file =  "tokenizer_{0}.json",
    experiment_name =  "runs/tmodel"