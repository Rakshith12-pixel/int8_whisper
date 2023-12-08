from typing import Any, Dict, List, Union
import evaluate
from transformers import WhisperForConditionalGeneration, GenerationConfig
from torch import nn
from torch.utils.data.dataloader import DataLoader
from transformers import AdamW, set_seed
import argparse
from accelerate import Accelerator
import bitsandbytes as bnb
from datasets import load_from_disk
import torch
from transformers import AutoTokenizer, AutoFeatureExtractor, AutoModel, AutoProcessor, LlamaTokenizer
import os
from os.path import dirname, abspath
from tqdm.auto import tqdm
from datasets import load_dataset, DatasetDict
import transformers, datasets
from transformers import WhisperFeatureExtractor
from transformers import WhisperTokenizer
from transformers import WhisperProcessor
from transformers import get_linear_schedule_with_warmup
from datasets import Audio
import torch
from dataclasses import dataclass
from typing import Any, Dict, List, Union
import evaluate
from transformers import WhisperForConditionalGeneration, GenerationConfig
from torch import nn
from torch.utils.data.dataloader import DataLoader
from transformers import AdamW, set_seed
import argparse
from accelerate import Accelerator
import bitsandbytes as bnb
from datasets import load_from_disk
import sys
from transformers import AutoTokenizer, AutoFeatureExtractor, AutoModel, AutoProcessor, LlamaTokenizer
import time
import bitsandbytes as bnb


#for i in range(model.config.num_hidden_layers):
#for layer in model.model.encoder.layers:
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
#linear_layers = []
#for layer in model.model.encoder.layers:
model=WhisperForConditionalGeneration.from_pretrained("/home/ujan/bitsandbytes/whisper-small_hindi_checkpoint")
model.to(device)
linear_layers = []
for name, layer in model.named_modules():
    if isinstance(layer, torch.nn.Linear) and "fc" in name:
        linear_layers.append(layer)
        

def optimize_8_bit(model):
    for linear_layer in linear_layers:
        layer_dim_in, layer_dim_out = linear_layer.weight.size()
        # Use bnb.nn.Linear8bitLt for 8-bit optimization
        optimized_layer = bnb.nn.Linear8bitLt(layer_dim_in, layer_dim_out, bias=(linear_layer.bias is not None))
        optimized_layer.weight.data = linear_layer.weight.data
        if linear_layer.bias is not None:
            optimized_layer.bias.data = linear_layer.bias.data
        # Replace the original linear layer with the optimized one
        setattr(model, name, optimized_layer)
#        print(model)
        model_opt=model
    return model_opt


model_opt_8=optimize_8_bit(model)
model_opt_8.to(device)
print(model_opt_8)

@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    """
    Data collator that will dynamically pad the inputs received.
    Args:
        processor ([`WhisperProcessor`])
            The processor used for processing the data.
        decoder_start_token_id (`int`)
            The begin-of-sentence of the decoder.
        forward_attention_mask (`bool`)
            Whether to return attention_mask.
    """

    processor: Any
    decoder_start_token_id: int
    forward_attention_mask: bool

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lengths and need
        # different padding methods
        model_input_name = self.processor.model_input_names[0]
        input_features = [{model_input_name: feature[model_input_name]} for feature in features]
        label_features = [{"input_ids": feature["labels"]} for feature in features]

        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        if self.forward_attention_mask:
            batch["attention_mask"] = torch.LongTensor([feature["attention_mask"] for feature in features])

        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        # if bos token is appended in previous tokenization step,
        # cut bos token here as it's append later anyways
        if (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
            labels = labels[:, 1:]
        #whisper small finetuned hindi

        #batch["decoder_input_ids"] = torch.tensor([[1,1]])*model.config.decoder_start_token_id
        batch["labels"] = labels

#        print("batch labels is {}".format(batch["labels"]))
#        inp_tar=[]
#        for _ in trange(10,desc="catching inputs",leave=False):
#            i = random.randint(0, batch["labels"].shape[1] - 20 - 1)
#            j = i + 20
#            inp = batch["labels"][:, i:j]
#            tar = inp.clone()
#            tar[:, :-1] = -100
#            inp_tar.append((inp,tar))
        #print("batch is {}".format(batch))
        return batch
        
def test():
    # extractor, tokenizer, processor
   # feature_extractor = AutoFeatureExtractor.from_pretrained("/home/ujan/speech-processing/models/whisper/whisper-small_Datasets")
   # tokenizer = AutoTokenizer.from_pretrained("/home/ujan/speech-processing/models/whisper/whisper-small_Datasets")
    # We only need to set the task id when the language is specified (i.e. in a multilingual setting)
    #tokenizer.set_prefix_tokens(language="hi", task="transcribe")
   # processor = AutoProcessor.from_pretrained("/home/ujan/speech-processing/models/whisper/whisper-small_Datasets")

    # model
    #model = AutoModel.from_pretrained("/home/ujan/distillation-asr/whisper-models/finetuned/whisper-small/hindi/checkpoint")
    feature_extractor = AutoFeatureExtractor.from_pretrained("/home/ujan/bitsandbytes/whisper-small_Datasets")
    tokenizer = AutoTokenizer.from_pretrained("/home/ujan/bitsandbytes/whisper-small_Datasets")

                # We only need to set the task id when the language is specified (i.e. in a multilingual setting)
    tokenizer.set_prefix_tokens(language="hi", task="transcribe")
                       # processor = WhisperProcessor.from_pretrained(args.model_name_or_path, language=args.model_lang, task=args.task)
    processor = AutoProcessor.from_pretrained("/home/ujan/bitsandbytes/whisper-small_Datasets")
    model_cfg=  AutoModel.from_pretrained("/home/ujan/bitsandbytes/openai_whisper-small")
   # model = AutoModel.from_pretrained("/home/ujan/distillation-asr/whisper-models/finetuned/whisper-small/chinese/checkpoint")
    #model.config.forced_decoder_ids = None
    model_cfg.config.forced_decoder_ids = processor.get_decoder_prompt_ids(language="hi", task="transcribe")
    model_cfg.config.suppress_tokens = []

    if model_cfg.config.decoder_start_token_id is None:
        raise ValueError("Make sure that `config.decoder_start_token_id` is correctly defined")


#    if args.freeze_encoder:
#        model.freeze_encoder()
#        model.model.encoder.gradient_checkpointing = False


    ## save config ##


    # dataset
    common_voice = DatasetDict()
    #--cv hindi--

    common_voice["train"] = load_from_disk("/home/ujan/common_voice_train")
    common_voice["test"] = load_from_disk("/home/ujan/common_voice_test")

#    common_voice["train"] = load_from_disk("/home/ujan/Datasets/final_train.csv")
#    common_voice["test"] = load_from_disk("/home/ujan/Datasets/final_test.csv")
        #--common voice chinese--
   # common_voice["train"] = load_from_disk("/media/ujan/MHST/distillation-asr/pytorch/CV-CN/train")
   # common_voice["test"] = load_from_disk("/media/ujan/MHST/distillation-asr/pytorch/CV-CN/test")

    training_data = common_voice["train"]
    print("training data is {}".format(training_data))
#    with accelerator.main_process_first():
        # remove unused columns
    common_voice = common_voice.remove_columns(["accent", "age", "client_id", "down_votes", "gender", "locale", "path", "segment", "up_votes"])

                                                                                                                                                                                         #385,1         53%

    common_voice["train"] = common_voice["train"].select(range(200)) #amount of data sent by get_loaders

    #if args.max_test_samples is not None:
    common_voice["test"] = common_voice["test"].select(range(300))
    print("common voice is {}".format(common_voice['test']))
     # resample to 16kHz
    common_voice = common_voice.cast_column("audio", Audio(sampling_rate=16000))


    # if SpecAugment is used for whisper models, return attention_mask to guide the mask along time axis
    forward_attention_mask = (
        getattr(model.config, "model_type", None) == "whisper"
        and getattr(model.config, "apply_spec_augment", False)
        and getattr(model.config, "mask_time_prob", 0) > 0
    )
  #231,0-1       38%
    max_input_length = 20 * feature_extractor.sampling_rate
    min_input_length = 0 * feature_extractor.sampling_rate
    #audio_column_name = args.audio_column_name
    #num_workers = args.num_workers
    #text_column_name = args.text_column_name
    #do_lower_case = args.do_lower_case
    model_input_name = feature_extractor.model_input_names[0]

    # function to vectorize dataset
    #def prepare_dataset(batch):
        # load and resample audio data from 48 to 16kHz
        #audio = batch["audio"]

        # compute log-Mel input features from input audio array 
        #features = feature_extractor(audio["array"], sampling_rate=audio["sampling_rate"], return_attention_mask=True)
        #batch["input_features"] = features.input_features[0]
        #batch["attention_mask"] = features.attention_mask[0]

        # encode target text to label ids 
        #batch["labels"] = tokenizer(batch["sentence"]).input_ids

        #return batch

    def prepare_dataset(batch):
        # process audio
        sample = batch["audio"]
        inputs = feature_extractor(
            sample["array"],
            sampling_rate=sample["sampling_rate"],
            return_attention_mask=forward_attention_mask
        )
        batch["input_features"] = inputs.input_features[0]
                                                                                                                                                                                         #493,1         60%

        # process audio length
        batch[model_input_name] = inputs.get(model_input_name)[0]
        batch["input_length"] = len(sample["array"])
        if forward_attention_mask:
            batch["attention_mask"] = inputs.get("attention_mask")[0]

        # process targets
        input_str = batch["sentence"]  # do lower
        batch["labels"] = tokenizer(input_str).input_ids
    #    print("batch labels is {}".format(batch['labels']))
        return batch


#    with accelerator.main_process_first():
        # vectorize dataset
    common_voice = common_voice.map(prepare_dataset,remove_columns=common_voice.column_names["test"])

    data_collator = DataCollatorSpeechSeq2SeqWithPadding(
        processor=processor,
        decoder_start_token_id=model.config.decoder_start_token_id,
        forward_attention_mask=forward_attention_mask,
    )
    # data loaders
    test_dataloader = DataLoader(
        common_voice["test"],
        shuffle=True,
        collate_fn=data_collator,
        batch_size=1,
    )
    return test_dataloader



def get_test_loaders():
  data = test()
  return data

print("\n============ Evaluating CER... ============")
@torch.no_grad()
def evaluate_cer(model):
  print("model size used in CER is {}".format(sys.getsizeof(model)))
  cer_metric = evaluate.load("/home/ujan/bitsandbytes/cer.py")
  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
  print("loading data for CER")
  test_dataloader = get_test_loaders()
            #args.dataset,
            #nsamples=args.nsamples,
            #seed=32,
            #seqlen=20,
            #eval_mode=True)
 # print(test_dataloader)
  print("number of data points are {}".format(len(test_dataloader.sampler)))
  tick = time.time()
      #train_dataloader = get_loaders(
      #  common_voice["train"],
      #  shuffle=True,
      #  collate_fn=data_collator,
      #  batch_size=args.train_batch_size)
      #model, train__dataloader = accelerator.prepare(model, eval_dataloader)
  time_quant=[]
  time_unquant=[]
  time_unquant_noiseless=[]
  i=1
  for batch in test_dataloader:
    with torch.no_grad():
         # print(type(model))
         # print(batch)
      batch.to(device)
      tick = time.time()
                                                                                                                                                                                         #407,14        43%

      outputs = model_opt_8(**batch)
      print(f"time for inference for a batch of data on model 8 bit quantized model is: {time.time() - tick:.1f}")
      time_quant.append(time.time()-tick)

      model2 = WhisperForConditionalGeneration.from_pretrained("/home/ujan/bitsandbytes/checkpoint")

      tick = time.time()

      outputs2 = model2(**batch)

      print(f"time for inference for a batch of data on unquantized model without quant noise is: {time.time() - tick:.1f}")
      time_unquant.append(time.time()-tick)


                    # compute metric
                    # generate and calculate cer, wer
                    ## slow ##
    output_ids = accelerator.unwrap_model(model).generate(batch["input_features"])
                    # pad_acrss_processes recursively pads the tensors 
                    # in a nested list/tuple/dictionary of tensors from all devices 
                    # to the same size so they can safely be gathered
    output_ids = accelerator.pad_across_processes(output_ids, dim=1, pad_index=tokenizer.pad_token_id)
    label_ids = accelerator.pad_across_processes(batch["labels"], dim=1, pad_index=tokenizer.pad_token_id)
                    # gather from all devices
    output_ids = accelerator.gather(output_ids)
    print("output_ids(quantized) are {}".format(output_ids))  #.cpu().numpy()  # gather_for_metrics
    label_ids = accelerator.gather(label_ids)  #.cpu().numpy()  # gather_for_metrics
                    # decode ids
    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
    predictions = processor.batch_decode(output_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
    print("predictions are {}".format(predictions))
    references = processor.batch_decode(
                        label_ids,
                        group_tokens=False,
                        skip_special_tokens=True,
                        clean_up_tokenization_spaces=True)
    print("refrences are {}".format(references))
    cer_metric.add_batch(predictions=predictions, references=references)
    print("batch number is {}".format(i))
    i=i+1

    cer_result = cer_metric.compute()

  print("CER(cer_result*100) value for quantized model is {}".format(cer_result*100))

evaluate_cer(model_opt_8)
device = "cuda:0" if torch.cuda.is_available else "cpu"
