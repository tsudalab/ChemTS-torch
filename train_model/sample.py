import yaml
from model.rnn import SmilesPredModule
from model.vocab import Tokenizer

model_conf = "model_setting.yaml"
with open(model_conf, "r") as f:
    model_conf = yaml.load(f, Loader=yaml.SafeLoader)
model_ckp = "../model/user_trained/smi_acc-epoch=113-val_acc=0.79.ckpt"
model = SmilesPredModule.load_from_checkpoint(model_ckp, conf=model_conf)

vocab_file = "../model/user_trained/tokens.txt"
tokenizer = Tokenizer.from_file(vocab_file)

model.rnn_model.eval()
samples = model.rnn_model.sample(10, bos=tokenizer.bos_id, eos=tokenizer.eos_id, device="cuda:3")
print(samples)
for sample in samples.cpu().numpy().tolist():
    print(tokenizer.int2smi(sample, remove_eos=True))