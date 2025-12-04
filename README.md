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

The next lines show a snippet of the *YAML* file defining the configuration of the ROS2 RAG node:

```yaml
ros2_rag:
  ros__parameters:
    generator_family: 'flan_t5'
    generator_version: 'small'
    generator_loading: 'pretrained'
    knowledge_base_path: '/home/ubuntu/knowledge_base'
```

Additionally, as ROS2 RAG is implemented as a lifecycle node, the *auto_activate* launch argument (by default *false*) allows defining if the node configures and activates automatically.

## ROS2 RAG Services

The ROS2 RAG node offers the next services:

* **/ros2_rag/query:** Service to query the RAG system, generating the completion using only the LLM.
* **/ros2_rag/rag_query:** Service to query the RAG system, creating an augmented query with information retrieved from the knowledge base. The service includes two parameters, the *query* and *query template*. This *query template* contains the complete prompt where the ROS2 RAG node will insert context retrieved from the knowledge base as well as the query itself. To this end, the *query template* **MUST** contain the labels **%context%** and **%query%** to identify the insertion points. Here is an snippet of a valid template:

```text
You are a helpful AI assistant.
Use the context below to answer the user question.
--- CONTEXT START ---
%context%
--- CONTEXT END ---
--- QUESTION ---
%query%
--- ANSWER ---
```

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
