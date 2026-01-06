# ROS2 RAG

<p>
  <a href="https://github.com/aitor-ibarguren/ros2_rag/actions/workflows/ros2_jazzy_ci.yml">
    <img src="https://github.com/aitor-ibarguren/ros2_rag/actions/workflows/ros2_jazzy_ci.yml/badge.svg" alt="Build">
  </a>
  <a href="https://github.com/aitor-ibarguren/ros2_rag/actions/workflows/isort.yml">
    <img src="https://github.com/aitor-ibarguren/ros2_rag/actions/workflows/isort.yml/badge.svg" alt="isort">
  </a>
  <a href="https://github.com/aitor-ibarguren/ros2_rag/actions/workflows/flake8_lint.yml">
    <img src="https://github.com/aitor-ibarguren/ros2_rag/actions/workflows/flake8_lint.yml/badge.svg" alt="isort">
  </a>
</p>

This repository contains the ROS2 RAG package, a ROS2 lifecycle node implementing a RAG (Retrieval-Augmented Generation) system. The package facilitates the deployment of a RAG system, offering a simple configuration through YAML files and a set of ROS2 services to query and manage the system.

Both the retriever and generator of the RAG system are implemented through the [llm_wrappers](https://github.com/aitor-ibarguren/llm_wrappers) repository, included as a Git submodule, which offers different Python classes to manage the system components. Specifically, these classes are based on [Transformers](https://github.com/huggingface/transformers/tree/main) and [FAISS](https://github.com/facebookresearch/faiss) libraries for the development of the generator and retriever modules.

Further information about the *ros2_rag* package can be found in the next sections:

- [Installation](#installation)
- [RAG System Configuration](#rag-system-configuration)
- [ROS2 Services](#ros2-services)
- [RAG System Features](#rag-system-features)
- [Dockerfile](#dockerfile)
- [License](#license)

## Installation

As mentioned previosly, the *ros2_rag* repository contains several base Python classes included as a Git submodule. Therefore, it is necessary to set the `--recursive` flag when cloning to ensure that submodules are retrieved:

```bash
# Clone repo & submodules
git clone --recursive https://github.com/aitor-ibarguren/ros2_rag.git
```

Additionally, it is also necessary to install the dependencies of *llm_wrappers* submodule. As Ubuntu 24.04 enforces PEP 668 (“externally managed environment”), the use of `--break-system-packages` pip installation flag is recommended:

```bash
pip install -r ./ros2_rag/ros2_rag/llw_wrappers/requirements.txt --break-system-packages
```

## RAG System Configuration

The current implementation allows the configuration of several parameters of the RAG system. Specifically the parameters are listed below:

* **generator_family:** Family of generator included in *Transformers* library.
* **generator_version:** Version of generator included in *Transformers* library.
* **generator_loading:** Loading procedure for the generator.
* **knowledge_base_path:** The path of the knowledge base. If the folder does not exists or is empty, the node will create a new index that can be stored in this folder or any other by means of a ROS2 service provided by the node.
* **retriever**
  * **top_k:** Number of fetched chunks in the retriever search.
  * **alpha:** Alpha to weigh semantic and keyword search in hybrid search (*alpha* for semantic search and *1 - alpha* for keyword search).
* **generator**
  * **max_new_tokens:** Maximum new tokens generated in the queries and RAG queries.
  * **temperature:** Temperature of the generation. Controls the randomness of the output from very deterministic (0.1) to high diversity (1.0), with a balanced randomness value of 0.7.
  * **top_p:** Value to control the token sampling. Usual values range from 1.0 (sampling from all tokens) to 0.8 (safe sampling), with a good balance between quality and creativity at 0.9.
* **history_active:** Boolean to define if history information is used in queries.
* **history**
  * **recent_interaction_number:** Number of interactions (query and completion) stored as recent interactions. These recent interactions will be inserted in queries when history is active.
  * **evicted_interaction_number:** Number of interactions (query and completion) stored as evicted interactions. These evicted interactions will be used to create the summary of the conversation.
* **rag_verification_active:** Boolean to define if the claims of the completion are verified against the retrieved chunks.
* **rag_verification**
  * **entailment_threshold:** The entailment threshold used in RAG verification to accept a completion claim. The threshold ranges from 0.0 to 1.0, with a balanced value of 0.75 (mid-high confidence).

The next lines show a snippet of the *YAML* file defining the configuration of the ROS2 RAG node:

```yaml
ros2_rag:
  ros__parameters:
    generator_family: 'qwen'
    generator_version: 'small'
    generator_loading: 'pretrained'
    knowledge_base_path: '/home/ubuntu/knowledge_base'
    retriever:
      top_k: 5
      alpha: 0.6
    generator:
      max_new_tokens: 75
      temperature: 0.5
      top_p: 0.9
    history_active: true
    history:
      recent_interaction_number: 2
      evicted_interaction_number: 4
    format:
      remove_bullets: true
    rag_verification_active: true
    rag_verification:
      entailment_threshold: 0.75
```

The complete list of LLM models and versions is depicted in the next table:

| Generator Family | Family tag | Generator versions & tags |
| :--- | :--- | :--- |
| **Deepseek** | `deepseek` |➤ DeepSeek R1 Distill Qwen 1.5B - `r1_distill_qwen_tiny`<br>➤ DeepSeek R1 Distill Qwen 7B - `r1_distill_qwen_base`<br>➤ DeepSeek R1 Distill Llama 8B - `r1_distill_llama_base`<br>➤ DeepSeek R1 Distill Qwen 14B - `r1_distill_qwen_large`<br>➤ DeepSeek R1 Distill Qwen 32B - `r1_distill_qwen_xl`<br>➤ DeepSeek R1 Distill Llama 70B - `r1_distill_llama_xl` |
| **Qwen** | `qwen` |➤ Qwen 2.5 0.5B Instruct - `xtiny`<br>➤ Qwen 2.5 1.5B Instruct - `tiny`<br>➤ Qwen 2.5 3B Instruc - `small`<br>➤ Qwen 2.5 7B Instruct - `base`<br>➤ Qwen 2.5 14B Instruct - `large`<br>➤ Qwen 2.5 72B Instruct - `xl` |

Additionally, as ROS2 RAG is implemented as a lifecycle node, the *auto_activate* launch argument (by default *false*) allows defining if the node configures and activates automatically, launching the node as:

```bash
ros2 launch ros2_rag ros2_rag auto_activate:=true
```

## ROS2 Services

### Data Management

The ROS2 RAG node offers the following services for managing the knowledge base of the RAG system:

* **/ros2_rag/load_csv_data:** Service to load data to the knowledge base from a CSV file. It is necessary to define the path to the CSV file, as well as the CSV column name/header (only this column's information will be extracted when parsing the CSV file). Although the service manages each CSV row as a chunk, it is possible to activate an additional chunking to divide each cells text.
* **/ros2_rag/load_pdf_data:** Service to load data to the knowledge base from PDF files contained in the provided folder path. The service chunks the PDFs content.
* **/ros2_rag/save_index:** Saves the index (knowledge base) in the provided folder path for future uses.

The data loading services include arguments to define the chunking parameters:

* *chunk_size:* Number of characters of the chunk.
* *chunk_overlap:* Number of characters overlapped between adjacent chunks.

### Queries

The ROS2 RAG node offers the following services for querying the RAG system:

* **/ros2_rag/query:** Service to query the RAG system, generating the completion using only the LLM.
* **/ros2_rag/rag_query:** Service to query the RAG system, creating an augmented query with information retrieved from the knowledge base. The service includes two parameters, the *query* and *query template*. This *query template* contains the complete prompt where the ROS2 RAG node will insert context retrieved from the knowledge base as well as the query itself. To this end, the *query template* **MUST** contain the labels **%context%** and **%query%** to identify the insertion points. Here is a snippet of a valid template:

    ```text
    You are a helpful AI assistant.
    Use the context below to answer the user question.
    If the answer is not contained in the context, say "The question can not be answered".
    Respond only with the answer.

    CONTEXT:
    %context%
      
    QUERY:
    %query%
    ANSWER:
    ```

    If conversation history is active, the template **can** contain the labels **%conversation_summary%** and **%last_user_queries%** to insert information about the conversation history (more information about the history management is provided in the next subsection). Here is a snippet of a valid template including conversation history:

    ```text
    You are a helpful AI assistant.

    CONVERSATION SUMMARY:
    %conversation_summary%

    LAST USER QUERIES:
    %last_user_queries%

    Use the context below to answer the user question.
    If the answer is not contained in the context, say "The question can not be answered".
    Respond only with the answer.

    CONTEXT:
    %context%
      
    QUERY:
    %query%
    ANSWER:
    ```

    The service also returns the context chunks obtained from the retriever as well as the removed claims (when RAG verificaton active) to improve interpretability and facilitate debugging.

Both services include arguments to facilitate the RAG system queries:

* *return_answer_only:* Removes the query text from the completion, returning only the answer to the query.
* *remove_incomplete_sentences:* Removes last incomplete sentence if the completion does not finish with a dot character.

## RAG System Features

### History

The *ros2_rag* node allows managing the conversation/interaction history and inject it in the user queries. This history management can be activated and configured through the configuration YAML file by means of the [previously described parameters](#rag-system-configuration).

The history information is divided into three main elements:

* **Recent interactions:** The last interactions (query and completions) are stored and injected when the history is active.
  * In **standard queries**, both queries and completions are inserted as a header, creating an augmented prompt.
  * In **RAG queries**, only queries are inserted to avoid shadowing the retriever data.
* **Evicted interactions:** As new interactions arrive, the oldest ones are moved to evicted interactions. These evicted interactions are used to create a summary of the conversation, inserting this information in a summarized and concise way.
* **Summary:** The summary of the conversation is stored as a list of user goals, constraints, and context information. The summarization process is carried out by the LLM model when evicted interactions are inserted, executed in a separate thread launched right after the completion of a standard or RAG query is returned.

### RAG Verification

The *ros2_rag* node provides a verification step in **RAG queries** to check the entailment between the completion claims (sentences) and the chunks obtained from the retriever. The aim is reducing hallucinations from the LLM, removing information made up or not present in the context chunks. This verification can be activated and configured through the configuration YAML file by means of the [previously described parameters](#rag-system-configuration).

Specifically, the node makes use of `cross-encoder/nli-deberta-v3-base` cross-encoder model to calculate the entailment, contradiction, and neutrality scores. The entailment score is used to verify the alignment between each claim and the chunks, ensuring that at least one of these chunks entails the sentence.

# Dockerfile

To facilitate the usage and deployment of the package, a Dockerfile is included. This Dockerfile generates a Docker image that includes all the dependencies of the `ros2_rag` package as well as the ROS2 Jazzy framework. This image can be generated using the *docker build* command inside the repository folder:

```bash
docker build -t ros2-rag:jazzy .
```

The Docker image can be executed in interactive mode to test the provided ROS2 lifecycle node within a containerized environment as:


```bash
docker run -it ros2-rag:jazzy bash
```

initializing the ROS2 Jazzy environment and the the *ros2_rag* workspace as:

```bash
source /opt/ros/jazzy/setup.bash
source /home/ubuntu/ros2_ws/install/local_setup.bash
```

## License

The *ros2_rag* repository has an Apache 2.0 license, as found in the [LICENSE](LICENSE) file.
