import { PDFLoader } from "langchain/document_loaders/fs/pdf";
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";
import { OpenAIEmbeddings } from "@langchain/openai";
import { PGVectorStore } from "@langchain/community/vectorstores/pgvector";
import { ChatOllama } from "@langchain/community/chat_models/ollama";
import { PromptTemplate } from "@langchain/core/prompts";

const llm = new ChatOllama({
  baseUrl: "http://localhost:11434",
  model: "llama2",
});

const embeddings = new OpenAIEmbeddings({
  openAIApiKey: process.env.OPENAI_API_KEY,
  batchSize: 512,
  modelName: "text-embedding-3-large",
  // Vectors with up to 2,000 dimensions can be indexed
  // https://github.com/pgvector/pgvector#hnsw
  dimensions: 2000,
});

const pgvectorConfig = {
  postgresConnectionOptions: {
    type: "postgres",
    host: "127.0.0.1",
    port: 5432,
    database: "langchain_cookbook",
  },
  tableName: "retrieval_augmented_generation",
  columns: {
    idColumnName: "id",
    vectorColumnName: "vector",
    contentColumnName: "content",
    metadataColumnName: "metadata",
  },
};

const vectorStore = await PGVectorStore.initialize(embeddings, pgvectorConfig);

const prepare = async () => {
  await vectorStore.client.query(
    "DROP TABLE IF EXISTS retrieval_augmented_generation;",
  );

  await vectorStore.client.query(
    "CREATE TABLE retrieval_augmented_generation (id uuid PRIMARY KEY DEFAULT uuid_generate_v4(), content text, metadata jsonb, vector vector(2000));",
  );

  console.log("Finished preparing database!");
};

const index = async (path) => {
  console.log("Indexing input PDF...");

  // Load and chunk contents of PDF file
  const loader = new PDFLoader(path);
  const docs = await loader.load(); // One document per page

  const splitter = new RecursiveCharacterTextSplitter({
    chunkSize: 100,
    chunkOverlap: 25,
  });
  const allSplits = await splitter.splitDocuments(docs);

  // Index chunks
  await vectorStore.addDocuments(allSplits);

  console.log("Finished indexing PDF!");
};

const retrieve = async ({ query }) => {
  const retrievedDocs = await vectorStore.similaritySearch(query);

  return retrievedDocs;
};

const generate = async ({ question, context }) => {
  const promptTemplate =
    PromptTemplate.fromTemplate(`Answer the question based only on the following context:
{context}

Question: {question}`);

  const messages = await promptTemplate.invoke({
    question: question,
    context: context,
  });

  const response = await llm.invoke(messages);

  return response;
};

// =============================================================================

await prepare();

await index("retrieval_augmented_generation/the_real_mother_goose.pdf");

const retrievedDocs = await retrieve({
  query: "Who kissed the girls and made them cry?",
});

const response = await generate({
  question: "Who kissed the girls and made them cry?",
  context: retrievedDocs.map((doc) => doc.pageContent).join("\n"),
});

console.log(response);

await vectorStore.end();
