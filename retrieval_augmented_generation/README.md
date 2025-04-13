# Retrieval Augmented Generation (RAG)

## Environment

- This script requires Node `20.10.0` to run.
- Environment variable `OPENAI_API_KEY`.
- Install Ollama.

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

Pull model `llama2`.

``` shell
$ ollama pull llama2
```

Install dependencies

``` shell
$ yarn install
```

## Run

- Run the script with `node retrieval_augmented_generation/index.mjs`.
