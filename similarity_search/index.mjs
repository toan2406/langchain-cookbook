import { OpenAIEmbeddings } from "@langchain/openai";
import { PGVectorStore } from "@langchain/community/vectorstores/pgvector";

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
  tableName: "similarity_search",
  columns: {
    idColumnName: "id",
    vectorColumnName: "vector",
    contentColumnName: "content",
    metadataColumnName: "metadata",
  },
};

const pgvectorStore = await PGVectorStore.initialize(
  embeddings,
  pgvectorConfig
);

await pgvectorStore.client.query("DELETE FROM similarity_search;");

console.log("Cleaned database!");

await pgvectorStore.addDocuments([
  { pageContent: "Software Engineer", metadata: { id: 1 } },
  { pageContent: "Front-end Developer", metadata: { id: 2 } },
  { pageContent: "Back-end Developer", metadata: { id: 3 } },
  { pageContent: "Project Manager", metadata: { id: 4 } },
]);

// Search using cosine distance by default
// https://github.com/langchain-ai/langchainjs/blob/5df74e3/libs/langchain-community/src/vectorstores/pgvector.ts#L423

const queryString = "frontend dev";
const results = await pgvectorStore.similaritySearchWithScore(queryString, 10);

console.log(`The nearest neighbors of "${queryString}" by cosine distance are:`);
console.log(results);

await pgvectorStore.end();
