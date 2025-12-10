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
- [Getting Started](#getting-started)
- [ROS2 RAG System Comfiguration](#ros2-rag-system-comnfiguration)
- [ROS2 RAG Services](#ros2-rag-services)
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

## ROS2 RAG System Configuration

The current implementation allows the configuration of several parameters of the RAG system. Specifically the parameters are listed below:

* **generator_family:** Family of generator included in *Transformers* library.
* **generator_version:** Version of generator included in *Transformers* library.
* **generator_loading:** Loading procedure for the generator.
* **knowledge_base_path:** The path of the knowledge base. If the folder does not exists or is empty, the node will create a new index that can be stored in this folder or any other by means of a ROS2 service provided by the node.
* **max_new_tokens:** Maximum new tokens generated in the queries and RAG queries.
* **temperature:** Temperature of the generation. Controls the randomness of the output from very deterministic (0.1) to high diversity (1.0), with a balanced randomness value of 0.7.
* **top_p:** Value to control the token sampling. Usual values range from 1.0 (sampling from all tokens) to 0.8 (safe sampling), with a good balance between quality and creativity at 0.9.

The next lines show a snippet of the *YAML* file defining the configuration of the ROS2 RAG node:

```yaml
ros2_rag:
  ros__parameters:
    generator_family: 'qwen'
    generator_version: 'small'
    generator_loading: 'pretrained'
    knowledge_base_path: '/home/ubuntu/knowledge_base'
    max_new_tokens: 50
    temperature: 0.5
    top_p: 0.9
```

The complete list of LLM models and versions is depicted in the next table:

| Generator Family | Family tag | Generator versions & tags |
| :--- | :--- | :--- |
| **Deepseek** | `deepseek` |➤ DeepSeek R1 Distill Qwen 1.5B - `r1_distill_qwen_tiny`<br>➤ DeepSeek R1 Distill Qwen 7B - `r1_distill_qwen_base`<br>➤ DeepSeek R1 Distill Llama 8B - `r1_distill_llama_base`<br>➤ DeepSeek R1 Distill Qwen 14B - `r1_distill_qwen_large`<br>➤ DeepSeek R1 Distill Qwen 32B - `r1_distill_qwen_xl`<br>➤ DeepSeek R1 Distill Llama 70B - `r1_distill_llama_xl` |
| **Flan T5** | `flan_t5` |➤ Flan T5 small - `small`<br>➤ Flan T5 base - `base`<br>➤ Flan T5 large - `large`<br>➤ Flan T5 XL - `xl`<br>➤ Flan T5 XXL - `xxl` |
| **Qwen** | `qwen` |➤ Qwen 2.5 0.5B Instruct - `xtiny`<br>➤ Qwen 2.5 1.5B Instruct - `tiny`<br>➤ Qwen 2.5 3B Instruc - `small`<br>➤ Qwen 2.5 7B Instruct - `base`<br>➤ Qwen 2.5 14B Instruct - `large`<br>➤ Qwen 2.5 72B Instruct - `xl` |

Additionally, as ROS2 RAG is implemented as a lifecycle node, the *auto_activate* launch argument (by default *false*) allows defining if the node configures and activates automatically, launching the node as:

```bash
ros2 launch ros2_rag ros2_rag auto_activate:=true
```

## ROS2 RAG Services

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
* **/ros2_rag/rag_query:** Service to query the RAG system, creating an augmented query with information retrieved from the knowledge base. The service includes two parameters, the *query* and *query template*. This *query template* contains the complete prompt where the ROS2 RAG node will insert context retrieved from the knowledge base as well as the query itself. To this end, the *query template* **MUST** contain the labels **%context%** and **%query%** to identify the insertion points. Here is an snippet of a valid template:

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

Both services include arguments to facilitate the RAG system queries:

* *return_only_answer:* Removes the query text from the completion, returning only the answer to the query.
* *remove_incomplete_sentences:* Removes last incomplete sentence if the completion does not finish with a dot character.

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

The *llm_wrappers* repository has an Apache 2.0 license, as found in the [LICENSE](LICENSE) file.
