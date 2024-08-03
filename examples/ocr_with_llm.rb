# https://platform.openai.com/docs/guides/vision

require "dotenv/load"
require "langchain"
require "openai"
require "llama_cpp"
require "base64"

path = "documents/article.png"
image_contet = File.read(path)

encoded_base64_image = Base64.encode64(image_contet)

# llm = Langchain::LLM::Ollama.new(default_options: {chat_completion_model_name: "llama3.1"})
# messages = [
#   {
#     role: "user",
#     content: "Extract the text from the image below and provide it without any additional explanations " \
#               "or introductory ephrases. " \
#               "Image in base64\n![image](data:image/png;base64,#{encoded_base64_image})"
#   }
# ]

llm = Langchain::LLM::OpenAI.new(api_key: ENV["OPENAI_API_KEY"], default_options: { chat_completion_model_name: "gpt-4o-mini" })

messages = [
  {
    role: "user",
    content: [
      {
        type: "text",
        text: "Extract the text from the image below and provide it without any additional explanations " \
              "or introductory phrases."
      },
      {
        type: "image_url",
        image_url: {
          url: "data:image/png;base64,#{encoded_base64_image}"
        }
      }
    ]
  }
]

llm.chat(messages: messages).chat_completion
