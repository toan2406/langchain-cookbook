import { OpenAIEmbeddings } from "@langchain/openai";

const embeddings = new OpenAIEmbeddings({
  openAIApiKey: process.env.OPENAI_API_KEY,
  batchSize: 512, // Default is 512. Max is 2048
  modelName: "text-embedding-3-large", // text-embedding-3-large returns embeddings of dimension 3072
});

const vectors = await embeddings.embedDocuments(["some text"]);

console.log(vectors[0]);
