from transformers import CLIPTokenizer

clip_tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32", 
                                               TOKENIZERS_PARALLELISM=False, clean_up_tokenization_spaces=True)
