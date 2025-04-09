# Langchain integration for Galaxia retriever


## Galaxia Knowledge Base

> Galaxia Knowledge Base is an integrated knowledge base and retrieval mechanism for RAG. In contrast to standard solution, it is based on Knowledge Graphs built using symbolic NLP and Knowledge Representation solutions. Provided texts are analysed and transformed into Graphs containing text, language and semantic information. This rich structure allows for retrieval that is based on semantic information, not on vector similarity/distance.

Implementing RAG using Galaxia involves first uploading your files to [Galaxia](https://beta.cloud.smabbler.com/home), analyzing them there and then building a model (knowledge graph). When the model is built, you can use `GalaxiaRetriever` to connect to the API and start retrieving.


## Installation
```

```

## Usage

```
from langchain_galaxia_retriever.retriever import GalaxiaRetriever
from langchain_core.callbacks.manager import RunManager

gr = GalaxiaRetriever(
    api_url="beta.api.smabbler.com",
    api_key="<key>",
    knowledge_base_id="<knowledge_base_id>",
)

rm = RunManager.get_noop_manager()
result = gr._get_relevant_documents('<test question>', rm)
print(result)
```
