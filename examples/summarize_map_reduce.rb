# https://python.langchain.com/v0.2/docs/tutorials/summarization/#map-reduce

require "dotenv/load"
require "langchain"
require "openai"
require "llama_cpp"

path = "documents/article.txt"
chunker = Langchain::Chunker::RecursiveText

data_loaded = Langchain::Loader.new(path, chunker: chunker).load
data_in_chunks = data_loaded.chunks(chunk_size: 512, chunk_overlap: 80)

def summarize_chunk_prompt(chunk_number, total_chunks, text)
  %(
    The following text is chunk #{chunk_number} of #{total_chunks} from the article.
    Please summarize the text.

    Original text:
    #{text}

    AI Summary:
  )
end

# llm = Langchain::LLM::OpenAI.new(api_key: ENV["OPENAI_API_KEY"])
llm = Langchain::LLM::Ollama.new(default_options: {chat_completion_model_name: "llama3.1"})

summaries = data_in_chunks.map.with_index(1) do |chunk, index|
  text = summarize_chunk_prompt(index, data_loaded.chunks.size, chunk.text)
  llm.chat(messages: [{role: "user", content: text}]).chat_completion
end

def final_summary_prompt(summaries:)
  %(
    The following are summarized chunks from an article.
    Please combine these summaries into one cohesive summary and provide the final summary.

    Summarized chunks:
    #{summaries}

    Final AI Summary
  )
end

joined_summaries = summaries.join("\n\n")
combine_summaries = final_summary_prompt(summaries: joined_summaries)

puts llm.chat(messages: [{role: "user", content: combine_summaries}]).chat_completion
