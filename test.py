# from ollama import chat
# from ollama import ChatResponse

# response: ChatResponse = chat(model='llama3.2:1b', messages=[
#   {
#     'role': 'user',
#     'content': 'Why is the sky blue?',
#   },
# ])
# print(response['message']['content'])
# # or access fields directly from the response object
# print(response.message.content)

import torch

if torch.cuda.is_available():
    print("CUDA GPU is available.")
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")
else:
    print("No CUDA GPU found.")
