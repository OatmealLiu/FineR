import torch
from transformers import Blip2Processor, Blip2ForConditionalGeneration
from .bot_toolbox import get_chat_log


BLIP2ZOO = {
    'FlanT5-XXL': 'Salesforce/blip2-flan-t5-xxl',
    'FlanT5-XL-COCO': 'Salesforce/blip2-flan-t5-xl-coco',
    'FlanT5-XL': 'Salesforce/blip2-flan-t5-xl',
    'OPT6.7B-COCO': 'Salesforce/blip2-opt-6.7b-coco',
    'OPT2.7B-COCO': 'Salesforce/blip2-opt-2.7b-coco',
    'OPT6.7B': 'Salesforce/blip2-opt-6.7b',
    'OPT2.7B': 'Salesforce/blip2-opt-2.7b',
}

ANSWER_INSTRUCTION = 'Answer given questions. If you are not sure about the answer, say you don\'t ' \
                     'know honestly. Don\'t imagine any contents that are not in the image.'

SUB_ANSWER_INSTRUCTION = 'Answer: '  # template following blip2 huggingface demo


def trim_answer(answer):
    answer = answer.split('Question:')[0].replace('\n', ' ').strip()
    return answer


class VQABot:
    def __init__(self, model_tag, device='cpu', device_id=0, bit8=False, max_answer_tokens=-1):
        # load BLIP-2 to a single gpu
        self.model_tag = model_tag
        self.model_name = "BLIP-2"
        self.max_answer_tokens = max_answer_tokens

        self.blip2_processor = Blip2Processor.from_pretrained(BLIP2ZOO[self.model_tag])
        if device == 'cpu':
            self.device = 'cpu'
            self.blip2 = Blip2ForConditionalGeneration.from_pretrained(BLIP2ZOO[self.model_tag])
        else:
            self.device = 'cuda:{}'.format(device_id)
            self.bit8 = bit8
            dtype = {'load_in_8bit': True} if self.bit8 else {'torch_dtype': torch.float16}
            self.blip2 = Blip2ForConditionalGeneration.from_pretrained(BLIP2ZOO[self.model_tag],
                                                                       device_map={'': int(device_id)},
                                                                       **dtype)

    def get_name(self):
        return self.model_name

    def __call_blip2(self, raw_image, prompt):
        if self.device == 'cpu':
            inputs = self.blip2_processor(raw_image, prompt, return_tensors="pt")
        else:
            inputs = self.blip2_processor(raw_image, prompt, return_tensors="pt").to(self.device, torch.float16)

        out = self.blip2.generate(**inputs,  max_new_tokens=self.max_answer_tokens) \
            if self.max_answer_tokens > 0 else self.blip2.generate(**inputs)

        reply = self.blip2_processor.decode(out[0], skip_special_tokens=True)
        return reply

    def answer_chat_log(self, raw_image, chat_log, n_blip2_context=-1):
        # prepare the context for blip2
        blip2_prompt = '\n'.join([ANSWER_INSTRUCTION,
                                  get_chat_log(chat_log['questions'],chat_log['answers'],
                                               last_n=n_blip2_context), SUB_ANSWER_INSTRUCTION]
                                 )

        reply = self.__call_blip2(raw_image, blip2_prompt)
        trimmed_reply = trim_answer(reply)
        return reply, trimmed_reply

    def tell_me_the_obj(self, raw_image, super_class, super_unit):
        std_prompt = f"Questions: What is the {super_unit} of the {super_class} in this photo? Answer:"
        # std_prompt = f"Questions: What is the name of the main object in this photo? Answer:"
        reply = self.__call_blip2(raw_image, std_prompt)
        trimmed_reply = trim_answer(reply)
        return reply, trimmed_reply

    def describe_attribute(self, raw_image, attr_prompt):
        reply = self.__call_blip2(raw_image, attr_prompt)
        trimmed_reply = trim_answer(reply)
        return reply, trimmed_reply

    def caption(self, raw_image):
        # starndard way to caption an image in the blip2 paper
        std_prompt = 'a photo of'
        reply = self.__call_blip2(raw_image, std_prompt)
        reply = reply.replace('\n', ' ').strip()  # trim caption
        return reply

    def call_llm(self, prompts):
        prompts_temp = self.blip2_processor(None, prompts, return_tensors="pt")
        input_ids = prompts_temp['input_ids'].to(self.device)
        attention_mask = prompts_temp['attention_mask'].to(self.device, torch.float16)

        prompts_embeds = self.blip2.language_model.get_input_embeddings()(input_ids)

        outputs = self.blip2.language_model.generate(
            inputs_embeds=prompts_embeds,
            attention_mask=attention_mask)

        outputs = self.blip2_processor.decode(outputs[0], skip_special_tokens=True)
        return outputs
