import torch
from transformers import AutoModelForCausalLM
from JRX.janus.models import MultiModalityCausalLM, VLChatProcessor
from JRX.janus.utils.io import load_pil_images
from PIL import Image
import os
import PIL.Image
import numpy as np

class JRX:
    def __init__(self, model_path="deepseek-ai/Janus-Pro-7B"):
        self.vl_chat_processor = VLChatProcessor.from_pretrained(model_path)
        self.tokenizer = self.vl_chat_processor.tokenizer
        self.vl_gpt = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)
        self.vl_gpt = self.vl_gpt.to(torch.bfloat16).cuda().eval()

    def inference(self, prompt, images=None):
        if images is None:
            images = [Image.new("RGB", (224, 224), color="black")]

        conversation = [
            {
                "role": "<|User|>",
                "content": f"<image_placeholder>\n{prompt}",
                "images": images,
            },
            {"role": "<|Assistant|>", "content": ""},
        ]

        pil_images = load_pil_images(conversation)
        prepare_inputs = self.vl_chat_processor(
            conversations=conversation, images=pil_images, force_batchify=True
        )

        for key, value in vars(prepare_inputs).items():
            try:
                tensor_value = value.to(self.vl_gpt.device)
                if isinstance(tensor_value, torch.Tensor):
                    if key == "input_ids":
                        tensor_value = tensor_value.to(torch.long)
                    elif key in ["attention_mask", "images_seq_mask", "images_emb_mask"]:
                        tensor_value = tensor_value.to(torch.bool)
                    else:
                        tensor_value = tensor_value.to(torch.bfloat16)
                setattr(prepare_inputs, key, tensor_value)
            except AttributeError:
                pass

        if hasattr(prepare_inputs, 'images'):
            prepare_inputs.images = [img.to(torch.bfloat16) for img in prepare_inputs.images]

        if hasattr(prepare_inputs, 'input'):
            prepare_inputs.input = prepare_inputs.input.to(torch.bfloat16)

        inputs_embeds = self.vl_gpt.prepare_inputs_embeds(**prepare_inputs)

        outputs = self.vl_gpt.language_model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=getattr(prepare_inputs, 'attention_mask'),
            pad_token_id=self.tokenizer.eos_token_id,
            bos_token_id=self.tokenizer.bos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            max_new_tokens=512,
            do_sample=False,
            use_cache=True,
        )

        answer = self.tokenizer.decode(outputs[0].cpu().tolist(), skip_special_tokens=True)
        return answer

    @torch.inference_mode()
    def txt2img(self, prompt, temperature=1, parallel_size=1, cfg_weight=5, image_token_num_per_image=576, img_size=384, patch_size=16):
        conversation = [{"role": "<|User|>", "content": prompt}, {"role": "<|Assistant|>", "content": ""}]
        sft_format = self.vl_chat_processor.apply_sft_template_for_multi_turn_prompts(
            conversations=conversation,
            sft_format=self.vl_chat_processor.sft_format,
            system_prompt="",
        )
        prompt = sft_format + self.vl_chat_processor.image_start_tag

        input_ids = self.vl_chat_processor.tokenizer.encode(prompt)
        input_ids = torch.LongTensor(input_ids)

        tokens = torch.zeros((parallel_size*2, len(input_ids)), dtype=torch.int).cuda()
        for i in range(parallel_size*2):
            tokens[i, :] = input_ids
            if i % 2 != 0:
                tokens[i, 1:-1] = self.vl_chat_processor.pad_id

        inputs_embeds = self.vl_gpt.language_model.get_input_embeddings()(tokens)

        generated_tokens = torch.zeros((parallel_size, image_token_num_per_image), dtype=torch.int).cuda()

        for i in range(image_token_num_per_image):
            outputs = self.vl_gpt.language_model.model(inputs_embeds=inputs_embeds, use_cache=True, past_key_values=outputs.past_key_values if i != 0 else None)
            hidden_states = outputs.last_hidden_state
            
            logits = self.vl_gpt.gen_head(hidden_states[:, -1, :])
            logit_cond = logits[0::2, :]
            logit_uncond = logits[1::2, :]
            
            logits = logit_uncond + cfg_weight * (logit_cond-logit_uncond)
            probs = torch.softmax(logits / temperature, dim=-1)

            next_token = torch.multinomial(probs, num_samples=1)
            generated_tokens[:, i] = next_token.squeeze(dim=-1)

            next_token = torch.cat([next_token.unsqueeze(dim=1), next_token.unsqueeze(dim=1)], dim=1).view(-1)
            img_embeds = self.vl_gpt.prepare_gen_img_embeds(next_token)
            inputs_embeds = img_embeds.unsqueeze(dim=1)

        dec = self.vl_gpt.gen_vision_model.decode_code(generated_tokens.to(dtype=torch.int), shape=[parallel_size, 8, img_size//patch_size, img_size//patch_size])
        dec = dec.to(torch.float32).cpu().numpy().transpose(0, 2, 3, 1)

        dec = np.clip((dec + 1) / 2 * 255, 0, 255)

        visual_img = np.zeros((parallel_size, img_size, img_size, 3), dtype=np.uint8)
        visual_img[:, :, :] = dec

        os.makedirs('generated_samples', exist_ok=True)
        for i in range(parallel_size):
            save_path = os.path.join('generated_samples', "img_{}.jpg".format(i))
            PIL.Image.fromarray(visual_img[i]).save(save_path)

        return visual_img

if __name__ == "__main__":
    m = JRX()
    result = m.inference(prompt="What color do you see?", images=None)
    print(result)

    images = m.txt2img(prompt="Create a hacker-inspired, solana, bitcoin, crypto-military themed abstract art background with a futuristic, edgy vibe, incorporating digital elements, binary code, and geometric patterns, fully vector-compatible for high-quality scaling.")
    for img in images:
        PIL.Image.fromarray(img).show()
