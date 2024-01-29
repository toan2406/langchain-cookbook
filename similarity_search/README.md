# Similarity Search

This script searches for similar texts from an input text by cosine distance and returns the results along with their scores. The following configs are used in the demo:

- OpenAI's `text-embedding-3-large` model for text embedding.
- `pgvector` for vector store.
- \[Optional\] `HNSW` algorithm for ANN (approximately nearest neighbors).

## Environment

- This script requires Node `20.10.0` to run.
- Environment variable `OPENAI_API_KEY`.

## Preparation

Install `pgvector`.

``` shell
$ brew install postgresql@14
$ brew install pgvector
```

Enable vector extension.

``` shell
langchain_cookbook=# CREATE EXTENSION vector;
```

Create table to store your search data.

``` shell
langchain_cookbook=# CREATE TABLE similarity_search (id uuid PRIMARY KEY DEFAULT uuid_generate_v4(), content text, metadata jsonb, vector vector(2000));
```

Create HNSW index for ANN search.

``` shell
langchain_cookbook=# CREATE INDEX ON similarity_search USING hnsw (vector vector_cosine_ops) WITH (m = 16, ef_construction = 64);
```

Install dependencies

``` shell
$ yarn install
```

## Run

- Update `similarity_search/index.mjs` with your data.
- Run the script with `node similarity_search/index.mjs`.
